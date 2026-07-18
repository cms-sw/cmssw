#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <alpaka/alpaka.hpp>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometryMayOwnPtr.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizerHelper.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthECLCCEpilogue.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthECLCCEpilogueMultiBlock.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthECLCCFinalizeEpilogue.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringCCLabelsHostCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthECLCCEpilogueArgsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthECLCCEpilogueArgsHostCollection.h"

#ifdef EPILOGUE_MULTIBLOCK
static constexpr bool multiblock = true;
#else
static constexpr bool multiblock = false;
#endif

#ifdef EPILOGUE_COOPERATIVE
static constexpr bool cooperative = true;
#else
static constexpr bool cooperative = false;
#endif

//T4:40 RTX:72 L40S:142 4090:128 5080:84

//static constexpr unsigned int nIters = 10000;
static constexpr int nColdIters = 10;

using PFRecHitsNeighbours = Eigen::Matrix<int32_t, 8, 1>;

using namespace reco;

static bool verbose = false;

// Simple axis-aligned test cell:
namespace {
  class BoxCell final : public CaloCellGeometry {
  public:
    enum ParIdx { kDx = 0, kDy = 1, kDz = 2, kNPar = 3 };

    BoxCell(CornersMgr *cMgr, GlobalPoint const &center, CCGFloat const *par) : CaloCellGeometry(center, cMgr, par) {
      initSpan();
    }

    void vocalCorners(Pt3DVec &out, CCGFloat const *, Pt3D &ref) const final {
      out.clear();
      out.reserve(k_cornerSize);
      auto const &cv = getCorners();
      for (unsigned i = 0; i < k_cornerSize; ++i)
        out.emplace_back(cv[i].x(), cv[i].y(), cv[i].z());
      auto const &P = getPosition();
      ref = Pt3D(P.x(), P.y(), P.z());
    }

    // compute 8 corners from (center, dx,dy,dz) stored in param()
    void initCorners(CornersVec &c) final {
      auto const *p = param();  // points to 3 floats: dx,dy,dz
      auto const dx = p[kDx], dy = p[kDy], dz = p[kDz];
      auto const &C = getPosition();

      // indices: 0..3 at z - dz, 4..7 at z + dz (match CaloCellGeometry::initBack)
      c[0] = GlobalPoint(C.x() - dx, C.y() - dy, C.z() - dz);
      c[1] = GlobalPoint(C.x() + dx, C.y() - dy, C.z() - dz);
      c[2] = GlobalPoint(C.x() + dx, C.y() + dy, C.z() - dz);
      c[3] = GlobalPoint(C.x() - dx, C.y() + dy, C.z() - dz);
      c[4] = GlobalPoint(C.x() - dx, C.y() - dy, C.z() + dz);
      c[5] = GlobalPoint(C.x() + dx, C.y() - dy, C.z() + dz);
      c[6] = GlobalPoint(C.x() + dx, C.y() + dy, C.z() + dz);
      c[7] = GlobalPoint(C.x() - dx, C.y() + dy, C.z() + dz);
    }
  };

  CaloCellGeometry::CornersMgr s_cornersMgr(8 * 1048576, CaloCellGeometry::k_cornerSize);  //k_cornerSize = 8;

  CaloCellGeometry::ParMgr s_parMgr(8 * 1048576, /*subSize=*/BoxCell::kNPar);

  CaloCellGeometry::ParVecVec s_parBlocks;

  const CaloCellGeometry::CCGFloat *s_boxParPtr = [] {
    using CCF = CaloCellGeometry::CCGFloat;
    std::vector<CCF> pars = {1.f, 1.f, 1.f};
    return CaloCellGeometry::getParmPtr(pars, &s_parMgr, s_parBlocks);
  }();

  CaloCellGeometryMayOwnPtr makeBoxCellGeo(float x, float y, float z) {
    auto base = std::make_unique<BoxCell>(&s_cornersMgr, GlobalPoint(x, y, z), s_boxParPtr);

    return CaloCellGeometryMayOwnPtr(std::move(base));
  }
}  // namespace

static inline HcalDetId makeDetId(int ieta, int iphi, int depth) {  //makeHBdetId

  iphi = std::clamp(iphi, 1, 72);
  ieta = std::clamp(ieta, 1, 16);

  depth = (depth <= 1) ? 1 : 2;

  return HcalDetId(HcalBarrel, ieta, iphi, depth);
}

static inline reco::PFRecHit makePFRecHit(
    PFLayer::Layer layer, const HcalDetId &detId, float energy, float x, float y, float z, uint32_t flags = 0) {
  // half-sizes for a simple cell; tweak as you like
  return reco::PFRecHit{makeBoxCellGeo(x, y, z), detId.rawId(), layer, energy, flags};
}

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class EpilogueTest {
    const int nStreams;
    const int nIters;
    const int threads;

  public:
    EpilogueTest(const int nStreams, const int nIters, const int threads)
        : nStreams(nStreams), nIters(nIters), threads(threads) {}

    void apply(std::vector<Queue> &queues,
               std::vector<reco::PFClusterDeviceCollection> &outPFCluster,
               std::vector<reco::PFRecHitFractionDeviceCollection> &outPFRecHitFracs,
               std::vector<reco::PFMultiDepthClusteringCCLabelsDeviceCollection> &mdpfClusteringVars,
               const std::vector<reco::PFClusterDeviceCollection> &pfClusters,
               const std::vector<reco::PFRecHitFractionDeviceCollection> &pfRecHitFracs,
               const std::vector<reco::PFRecHitDeviceCollection> &pfRecHit) const;
  };

  void EpilogueTest::apply(std::vector<Queue> &queues,
                           std::vector<reco::PFClusterDeviceCollection> &outPFCluster,
                           std::vector<reco::PFRecHitFractionDeviceCollection> &outPFRecHitFracs,
                           std::vector<reco::PFMultiDepthClusteringCCLabelsDeviceCollection> &mdpfClusteringVars,
                           const std::vector<reco::PFClusterDeviceCollection> &pfClusters,
                           const std::vector<reco::PFRecHitFractionDeviceCollection> &pfRecHitFracs,
                           const std::vector<reco::PFRecHitDeviceCollection> &pfRecHit) const {
    uint32_t items = std::is_same_v<Device, alpaka::DevCpu> ? 1 : threads;

    auto n = static_cast<uint32_t>(mdpfClusteringVars[0]->metadata().size());
    uint32_t groups = cms::alpakatools::divide_up_by(n, items);

    if (groups < 1) {
      printf("Skip kernel launch...\n");
      return;
    } else {
      printf("Number of groups :: %d\n", groups);
    }

    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);

    if constexpr (cooperative) {
      printf("Running epilogue in cooperative mode.\n");
    } else {
      printf("Running epilogue in noncooperative mode.\n");
    }

    std::vector<reco::PFMultiDepthECLCCEpilogueArgsDeviceCollection> devClusteringEpilogueArgs;
    devClusteringEpilogueArgs.reserve(nStreams);

    for (int s = 0; s < nStreams; s++) {
      devClusteringEpilogueArgs.emplace_back(reco::PFMultiDepthECLCCEpilogueArgsDeviceCollection(queues[s], n));
      alpaka::wait(queues[s]);
    }

    double wall_time = 0.0;
    for (int i = 0; i < nIters; i++) {
      auto wall_start = std::chrono::high_resolution_clock::now();

      if constexpr (multiblock) {
        for (int s = 0; s < nStreams; s++) {
          devClusteringEpilogueArgs[s].zeroInitialise(queues[s]);

          alpaka::exec<Acc1D>(queues[s],
                              workDiv,
                              ECLCCEpilogueRecHitFracOffsetsKernel{},
                              devClusteringEpilogueArgs[s].view(),
                              mdpfClusteringVars[s].view(),
                              pfClusters[s].view());

          alpaka::exec<Acc1D>(queues[s],
                              workDiv,
                              ECLCCEpilogueCCOffsetsKernel{},
                              outPFCluster[s].view(),
                              devClusteringEpilogueArgs[s].view(),
                              mdpfClusteringVars[s].view(),
                              pfClusters[s].view());

          alpaka::exec<Acc1D>(queues[s],
                              workDiv,
                              ECLCCFinalizeEpilogueKernel<32, cooperative>{},
                              outPFCluster[s].view(),
                              outPFRecHitFracs[s].view(),
                              devClusteringEpilogueArgs[s].view(),
                              mdpfClusteringVars[s].view(),
                              pfClusters[s].view(),
                              pfRecHitFracs[s].view(),
                              pfRecHit[s].view());

          alpaka::exec<Acc1D>(
              queues[s], workDiv, ECLCCLoadSeedsKernel{}, outPFCluster[s].view(), devClusteringEpilogueArgs[s].view());
        }  //nStreams
      } else {
        if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
          for (int s = 0; s < nStreams; s++) {
            alpaka::exec<Acc1D>(queues[s],
                                workDiv,
                                ECLCCEpilogueNaiveKernel{},
                                outPFCluster[s].view(),
                                outPFRecHitFracs[s].view(),
                                mdpfClusteringVars[s].view(),
                                pfClusters[s].view(),
                                pfRecHitFracs[s].view(),
                                pfRecHit[s].view());
          }
        } else {
          for (int s = 0; s < nStreams; s++) {
            alpaka::exec<Acc1D>(queues[s],
                                workDiv,
                                ECLCCEpilogueKernel<32, cooperative>{},
                                outPFCluster[s].view(),
                                outPFRecHitFracs[s].view(),
                                mdpfClusteringVars[s].view(),
                                pfClusters[s].view(),
                                pfRecHitFracs[s].view(),
                                pfRecHit[s].view());
          }
        }
      }  //end multi-block

      for (int s = 0; s < nStreams; s++)
        alpaka::wait(queues[s]);

      auto wall_stop = std::chrono::high_resolution_clock::now();
      //
      auto wall_diff = wall_stop - wall_start;

      //printf("Current wall time %10e\n", static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6);

      if (i > nColdIters)
        wall_time +=
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
    }

    unsigned int measIters = nIters - nColdIters;
    const auto walltime_per_iter = wall_time / measIters;
    const auto throughput = static_cast<double>(nStreams) / walltime_per_iter;
    printf("Wall time: %f sec per iter, throuput %6e events per second\n", wall_time / measIters, throughput);
  }

  void launch_epilogue_test(std::vector<Queue> &queues,
                            std::vector<reco::PFClusterHostCollection> &outClusters,
                            std::vector<reco::PFRecHitFractionHostCollection> &outRecHitFracs,
                            const std::vector<::reco::PFMultiDepthClusteringCCLabelsHostCollection> &hostClusteringVars,
                            const std::vector<::reco::PFClusterHostCollection> &hostClusters,
                            const std::vector<::reco::PFRecHitHostCollection> &hostRecHits,
                            const std::vector<::reco::PFRecHitFractionHostCollection> &hostRecHitFracs,
                            const int nStreams,
                            const int nIters,
                            const int threadsPerBlock) {
    EpilogueTest epilogue_test(nStreams, nIters, threadsPerBlock);

    auto hOutClusters = outClusters[0].view();
    auto hClusters = hostClusters[0].view();
    auto hRecHits = hostRecHits[0].view();

    const int nClusters = hClusters.size();
    const int nFracs = hClusters.nRHFracs();
    const int nHits = hRecHits.size();
    const int nOutClusters = hOutClusters.size();

    std::vector<reco::PFClusterDeviceCollection> outDevClusters;
    outDevClusters.reserve(nStreams);
    std::vector<reco::PFRecHitFractionDeviceCollection> outDevRecHitFracs;
    outDevRecHitFracs.reserve(nStreams);

    std::vector<reco::PFClusterDeviceCollection> devClusters;
    devClusters.reserve(nStreams);
    std::vector<reco::PFRecHitDeviceCollection> devRecHits;
    devRecHits.reserve(nStreams);
    std::vector<reco::PFRecHitFractionDeviceCollection> devRecHitFracs;
    devRecHitFracs.reserve(nStreams);

    std::vector<reco::PFMultiDepthClusteringCCLabelsDeviceCollection> devClusteringVars;
    devClusteringVars.reserve(nStreams);

    for (int s = 0; s < nStreams; s++) {
      auto dev_clusters = reco::PFClusterDeviceCollection{queues[s], nClusters};
      auto dev_hits = reco::PFRecHitDeviceCollection{queues[s], nHits};
      auto dev_rhfrac = reco::PFRecHitFractionDeviceCollection{queues[s], nFracs};
      auto dev_clustering_vars = reco::PFMultiDepthClusteringCCLabelsDeviceCollection{queues[s], nClusters};
      //
      alpaka::memcpy(queues[s], dev_clusters.buffer(), hostClusters[s].buffer());
      alpaka::memcpy(queues[s], dev_hits.buffer(), hostRecHits[s].buffer());
      alpaka::memcpy(queues[s], dev_rhfrac.buffer(), hostRecHitFracs[s].buffer());
      alpaka::memcpy(queues[s], dev_clustering_vars.buffer(), hostClusteringVars[s].buffer());

      devClusters.emplace_back(std::move(dev_clusters));
      devRecHits.emplace_back(std::move(dev_hits));
      devRecHitFracs.emplace_back(std::move(dev_rhfrac));
      devClusteringVars.emplace_back(std::move(dev_clustering_vars));

      outDevClusters.emplace_back(reco::PFClusterDeviceCollection(queues[s], nOutClusters));
      outDevRecHitFracs.emplace_back(reco::PFRecHitFractionDeviceCollection(queues[s], nFracs));
    }

    printf("Run epilogue kernel %d %d %d \n", nClusters, nHits, nFracs);
    epilogue_test.apply(
        queues, outDevClusters, outDevRecHitFracs, devClusteringVars, devClusters, devRecHitFracs, devRecHits);
    printf("...done.\n");

    alpaka::memcpy(queues[0], outClusters[0].buffer(), outDevClusters[0].buffer());
    alpaka::memcpy(queues[0], outRecHitFracs[0].buffer(), outDevRecHitFracs[0].buffer());

    for (int s = 0; s < nStreams; s++)
      alpaka::wait(queues[s]);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

struct Fraction {  // content is same as for SoA
  int pfrhIdx;     // index into hits[]
  int pfcIdx;      // index into clusters[]
  float frac;      // contribution of that hit to that cluster
};

std::pair<int, int> create(::reco::PFClusterCollection &clusters,
                           ::reco::PFRecHitCollection &hits,
                           std::vector<Fraction> &rhfracs,
                           std::vector<int> &seedIdx,
                           const int nClusters,
                           const int minHitsPerCluster = 2,
                           const int maxHitsPerCluster = 10,
                           const float shareProbability = 0.75f) {
  std::mt19937 rng(12345);

  // E in [0.5, 120], r in [174, 180], phi in [0, 0.25], z = 0
  std::uniform_real_distribution<double> energies_dist(0.5, 120.0);
  std::uniform_real_distribution<double> r_dist(174.0, 180.0);
  std::uniform_real_distribution<double> phi_dist(0.0, 0.25);
  std::uniform_real_distribution<double> z_dist(0.0, 0.5);

  // hits / fractions
  std::uniform_int_distribution<int> nRH_distr(minHitsPerCluster, maxHitsPerCluster);
  std::uniform_real_distribution<float> wRaw_distr(0.05f, 1.0f);
  std::uniform_real_distribution<float> dxy_distr(-2.5f, 2.5f);
  std::uniform_real_distribution<float> uni_distr(0.f, 1.f);

  std::vector<int> hitOwners;  // same length as hits: owner cluster index
  hitOwners.reserve(nClusters * maxHitsPerCluster);

  int nHits = 0;

  int tot_offset = 0;

  for (int i = 0; i < nClusters; ++i) {
    const bool is_depth1 = (i % 2 == 0);
    const PFLayer::Layer layer = is_depth1 ? PFLayer::Layer::HCAL_BARREL1 : PFLayer::Layer::HCAL_BARREL2;
    const double depth = is_depth1 ? 1. : 2.;

    const double energy = energies_dist(rng);

    const double r = r_dist(rng);
    const double phi = phi_dist(rng);

    const double x = r * std::cos(phi);
    const double y = r * std::sin(phi);
    const double z = z_dist(rng);

    ::reco::PFCluster cluster(layer, energy, x, y, z);
    cluster.setDepth(depth);

    const int nRH = nRH_distr(rng);

    nHits += nRH;

    std::vector<float> w(nRH);
    double sumW = 0.;

    for (int j = 0; j < nRH; ++j) {
      w[j] = wRaw_distr(rng);
      sumW += static_cast<double>(w[j]);
    }
    for (int j = 0; j < nRH; ++j) {
      w[j] /= sumW;
    }

    const double phc = std::atan2(y, x);
    int iphi = 1 + int((phc < 0 ? phc + 2 * M_PI : phc) / (2 * M_PI) * 72.0);
    int ieta_mag = std::max(1, std::min(16, int(std::hypot(y, x) / 11.0)));
    int ieta = (std::uniform_int_distribution<int>(0, 1)(rng) ? +ieta_mag : -ieta_mag);

    float best_energy = 0.f;
    int seed_idx = -1;

    for (int j = 0; j < nRH; ++j) {
      const float hx = x + dxy_distr(rng);
      const float hy = y + dxy_distr(rng);
      const float hz = z;
      const float eH = float(cluster.energy()) * w[j];

      const HcalDetId detId = makeDetId(ieta, ((iphi + j) % 72) + 1, depth);
      const int hIdx = tot_offset + j;
      auto hit = makePFRecHit(layer, detId, eH, hx, hy, hz);

      hitOwners.push_back(i);

      // primary fraction entry: this hit contributes to its owner cluster:
      const float frac_value = (cluster.energy() > 0.0) ? (eH / float(cluster.energy())) : 0.f;

      rhfracs.push_back(Fraction{hIdx, i, frac_value});

      float current_energy = frac_value * /*hit.energy()*/ eH;

      if (current_energy > best_energy) {
        best_energy = current_energy;
        seed_idx = hIdx;
      }

      hits.emplace_back(hit);
    }

    tot_offset += nRH;

    seedIdx[i] = seed_idx;

    if (seed_idx >= 0)
      cluster.setSeed(hits[seed_idx].detId());
    else
      cluster.setSeed(DetId(0));

    clusters.emplace_back(std::move(cluster));
  }

  int nFracs = nHits;

  for (int i = 0; i < nHits; i++) {
    if (uni_distr(rng) > shareProbability)
      continue;

    const int owner_idx = hitOwners[i];
    const int neigh_idx = (owner_idx + 1) % nClusters;  // simple neighbor

    // find the existing owner fraction entry (last push for that hit is fine but we search robustly)
    float ownerFrac = 0.f;
    int ownerEntry = -1;

    for (int j = nFracs - 1; j >= 0; --j) {
      if (rhfracs[j].pfrhIdx == i && rhfracs[j].pfcIdx == owner_idx) {
        ownerFrac = rhfracs[j].frac;
        ownerEntry = j;
        break;
      }
    }

    if (ownerEntry < 0)
      continue;
    const float share_frac = ownerFrac * (0.2f + 0.6f * uni_distr(rng));

    rhfracs.push_back(Fraction{i, neigh_idx, share_frac});
    rhfracs[ownerEntry].frac -= share_frac;

    if (rhfracs[ownerEntry].frac < 0.f)
      rhfracs[ownerEntry].frac = 0.f;
    ++nFracs;
  }

  if (verbose)
    printf("Generated cluster/rechit collections with %d hits and %d rechit fractions.\n", nHits, nFracs);

  return std::make_pair(nHits, nFracs);
}

void load(::reco::PFClusterHostCollection &hostClusters,
          ::reco::PFRecHitHostCollection &hostRecHits,
          ::reco::PFRecHitFractionHostCollection &hostRecHitFracs,
          const ::reco::PFClusterCollection &clusters,
          const ::reco::PFRecHitCollection &hits,
          const std::vector<Fraction> &rhfracs,
          const std::vector<int> &seedIdx) {
  auto hClusters = hostClusters.view();
  auto hRecHits = hostRecHits.view();
  auto hRecHitFracs = hostRecHitFracs.view();

  const int nClusters = hClusters.size();
  const int nHits = hRecHits.size();
  const int nFracs = hClusters.nRHFracs();

  for (int i = 0; i < nHits; ++i) {
    const ::reco::PFRecHit &rhit = hits[i];

    const auto &rhpos = rhit.position();

    hRecHits[i].detId() = rhit.detId();
    hRecHits[i].denseId() = i;  //fake parameter for now.
    hRecHits[i].energy() = rhit.energy();
    hRecHits[i].time() = 0.f;
    hRecHits[i].depth() = rhit.depth();
    hRecHits[i].layer() = rhit.layer();
    hRecHits[i].neighbours() = PFRecHitsNeighbours::Zero();
    hRecHits[i].x() = rhpos.x();
    hRecHits[i].y() = rhpos.y();
    hRecHits[i].z() = rhpos.z();
  }

  int recHitFracIdx = 0;

  for (int i = 0; i < nClusters; ++i) {
    const ::reco::PFCluster &cluster = clusters[i];

    const int recHitOffset = recHitFracIdx;

    int recHitFracSize = 0;
    for (int j = 0; j < nFracs; ++j) {
      if (rhfracs[j].pfcIdx == i) {
        hRecHitFracs[recHitFracIdx].frac() = rhfracs[j].frac;
        hRecHitFracs[recHitFracIdx].pfrhIdx() = rhfracs[j].pfrhIdx;
        hRecHitFracs[recHitFracIdx].pfcIdx() = i;
        ++recHitFracIdx;
        ++recHitFracSize;
      }
    }

    const auto &cpos = cluster.position();

    hClusters[i].depth() = cluster.depth();
    hClusters[i].seedRHIdx() = seedIdx[i];
    hClusters[i].topoId() = 0;
    hClusters[i].rhfracSize() = recHitFracSize;
    hClusters[i].rhfracOffset() = recHitOffset;
    hClusters[i].energy() = cluster.energy();
    hClusters[i].x() = cpos.x();
    hClusters[i].y() = cpos.y();
    hClusters[i].z() = cpos.z();
    hClusters[i].topoRHCount() = 0;  //?
    //printf("\n CLUSTER :: %u %u %f %u \n", recHitOffset, recHitFracSize, cluster.energy(), seedIdx[i]  );
  }
}

void create_cc_list(::reco::PFMultiDepthClusteringCCLabelsHostCollection &hostClusteringVars,
                    const std::vector<int> cc_roots,
                    const int nClusters) {
  auto hClusteringVars = hostClusteringVars.view();

  const int low_cc_roots_num = 2;  //zero and the next (after zero) cc roots
  const int cc_roots_num = static_cast<int>(cc_roots.size());

  std::mt19937 rng(12435);

  std::bernoulli_distribution pick_next_to_zero_cc_root(0.5);
  std::bernoulli_distribution pick_zero_cc_root_cond(0.1 / 0.5);

  std::uniform_int_distribution<int> pick_higher_cc_roots(low_cc_roots_num, cc_roots_num - 1);

  int current_cc_root_idx = 0;

  for (int i = 0; i < nClusters; ++i) {
    const bool is_cc_root = std::binary_search(cc_roots.begin(), cc_roots.end(), i);

    int x;
    if (is_cc_root) {
      x = i;
      if (current_cc_root_idx < cc_roots_num - 1)
        current_cc_root_idx += 1;
    } else {
      bool is_set = false;
      while (is_set == false) {
        if (pick_next_to_zero_cc_root(rng)) {
          x = 1;
          is_set = true;
        } else if (pick_zero_cc_root_cond(rng)) {
          x = 0;
          is_set = true;
        } else {
          x = cc_roots[pick_higher_cc_roots(rng)];
          if (x < cc_roots[current_cc_root_idx])
            is_set = true;
        }
      }
    }
    hClusteringVars[i].mdpf_topoId() = x;
  }
}

int checkEpilogue(Queue &queue,
                  const ::reco::PFClusterHostCollection &outHostClusters,
                  const ::reco::PFRecHitFractionHostCollection &outHostRecHitsFracs,
                  const ::reco::PFClusterHostCollection &inHostClusters,
                  const ::reco::PFRecHitHostCollection &recHits,
                  const ::reco::PFRecHitFractionHostCollection &inHostRecHitsFracs,
                  const ::reco::PFMultiDepthClusteringCCLabelsHostCollection &hostClusteringVars,
                  const std::vector<int> cc_roots) {
  auto hostClusteringVarsView = hostClusteringVars.view();
  auto inHostClustersView = inHostClusters.view();
  auto outHostClustersView = outHostClusters.view();
  auto inHostRecHitsFracsView = inHostRecHitsFracs.view();
  auto outHostRecHitsFracsView = outHostRecHitsFracs.view();
  auto recHitsView = recHits.view();

  const int inClustersNum = inHostClustersView.size();
  const unsigned int nVertices = hostClusteringVarsView.size();

  int nerrors = 0;

  const unsigned int nComponents = static_cast<unsigned int>(cc_roots.size());
  const unsigned int nFracs = outHostClustersView.nRHFracs();

  ::reco::PFClusterHostCollection outCheckHostClusters{queue, nComponents};
  ::reco::PFRecHitFractionHostCollection outCheckHostRecHitFracs{queue, nFracs};

  auto outPFCluster = outCheckHostClusters.view();
  auto outPFRecHitFracs = outCheckHostRecHitFracs.view();

  outPFCluster.nTopos() = nComponents;
  outPFCluster.nSeeds() = nComponents;
  outPFCluster.nRHFracs() = inHostClustersView.nRHFracs();
  outPFCluster.size() = nComponents;

  unsigned int ccrhfrac_idx = 0;
  unsigned int cc_idx = 0;

  for (auto &rep_idx : cc_roots) {
    outPFCluster[cc_idx].depth() = inHostClustersView[rep_idx].depth();
    outPFCluster[cc_idx].topoId() = cc_idx;
    outPFCluster[cc_idx].energy() = inHostClustersView[rep_idx].energy();
    outPFCluster[cc_idx].x() = inHostClustersView[rep_idx].x();
    outPFCluster[cc_idx].y() = inHostClustersView[rep_idx].y();
    outPFCluster[cc_idx].z() = inHostClustersView[rep_idx].z();
    outPFCluster[cc_idx].topoRHCount() = inHostClustersView[rep_idx].topoRHCount();

    // Setup rechitfracs:
    int cc_seed = inHostClustersView[rep_idx].seedRHIdx();
    float cc_energy = recHitsView[cc_seed].energy();

    outPFCluster[cc_idx].rhfracOffset() = ccrhfrac_idx;

    for (unsigned int iter_idx = 0; iter_idx < nVertices; iter_idx++) {
      const unsigned int comp_id = hostClusteringVarsView[iter_idx].mdpf_topoId();

      if (comp_id != static_cast<unsigned int>(rep_idx))
        continue;

      const int seed = inHostClustersView[iter_idx].seedRHIdx();
      const float energy = recHitsView[seed].energy();

      const unsigned int rhf_begin = inHostClustersView[iter_idx].rhfracOffset();
      const unsigned int rhf_end = rhf_begin + inHostClustersView[iter_idx].rhfracSize();

      for (unsigned int src_rhfrac_idx = rhf_begin; src_rhfrac_idx < rhf_end; src_rhfrac_idx++) {
        const unsigned int dst_rhfrac_idx = ccrhfrac_idx;

        outPFRecHitFracs[dst_rhfrac_idx].frac() = inHostRecHitsFracsView[src_rhfrac_idx].frac();
        outPFRecHitFracs[dst_rhfrac_idx].pfrhIdx() = inHostRecHitsFracsView[src_rhfrac_idx].pfrhIdx();
        outPFRecHitFracs[dst_rhfrac_idx].pfcIdx() = cc_idx;

        ccrhfrac_idx += 1;
      }
      //
      if (energy > cc_energy) {
        cc_energy = energy;
        cc_seed = seed;
      }
    }  // iter_idx

    outPFCluster[cc_idx].rhfracSize() = ccrhfrac_idx - outPFCluster[cc_idx].rhfracOffset();
    outPFCluster[cc_idx].seedRHIdx() = cc_seed;

    cc_idx += 1;
  }

  for (auto &cc_root : cc_roots) {
    if (verbose)
      printf("\n\n>>>>> Process cc root %d\n", cc_root);
    int tot_recFracSize = 0;
    bool is_valid_seed = false;
    int cc_seed = inHostClustersView[cc_root].seedRHIdx();
    float cc_energy = recHitsView[cc_seed].energy();

    for (int i = 0; i < inClustersNum; i++) {
      const int rep = hostClusteringVarsView[i].mdpf_topoId();
      if (rep == cc_root) {
        const auto recFracOffset = inHostClustersView[i].rhfracOffset();
        const auto recFracSize = inHostClustersView[i].rhfracSize();

        auto seed = inHostClustersView[i].seedRHIdx();

        auto seed_energy = recHitsView[seed].energy();

        tot_recFracSize += recFracSize;
        for (int j = 0; j < recFracSize; j++) {
          auto recHitidx = inHostRecHitsFracsView[recFracOffset + j].pfrhIdx();
          if (seed == recHitidx)
            is_valid_seed = true;
        }

        if (is_valid_seed == false) {
          nerrors += 1;
          if (verbose)
            printf("Component seed is not valid for cluster %d!\n", rep);
        } else {
          if (cc_energy < seed_energy) {
            cc_seed = seed;
            cc_energy = seed_energy;
            if (verbose)
              printf("updated seed for cluster %d\n", rep);
          }
          is_valid_seed = false;
        }
      }
    }
    if (verbose)
      printf("TOT rh size = %d , seed = %d (energy %f)\n", tot_recFracSize, cc_seed, cc_energy);
  }

  const int outClustersNum = outHostClustersView.size();

  for (int i = 0; i < outClustersNum; i++) {
    const auto recFracOffset = outHostClustersView[i].rhfracOffset();
    const auto recFracSize = outHostClustersView[i].rhfracSize();
    const auto clidx = outHostRecHitsFracsView[recFracOffset].pfcIdx();

    const auto seed = outHostClustersView[i].seedRHIdx();
    const auto energy = recHitsView[seed].energy();
    if (verbose)
      printf("OUT cluster RHF offset %d RHF size %d  idx %d seed %d energy %f\n",
             recFracOffset,
             recFracSize,
             clidx,
             seed,
             energy);
  }

  for (int i = 0; i < outClustersNum; i++) {
    const auto recFracOffset = outHostClustersView[i].rhfracOffset();
    const auto recFracSize = outHostClustersView[i].rhfracSize();
    const auto clidx = outHostRecHitsFracsView[recFracOffset].pfcIdx();
    const auto seed = outHostClustersView[i].seedRHIdx();

    bool match = false;
    unsigned int found_idx = nComponents;
    int nmatch = 0;
    for (int j = 0; j < static_cast<int>(nComponents); j++) {
      const auto test_seed = outPFCluster[j].seedRHIdx();

      if (test_seed == seed) {
        const auto test_recFracSize = outPFCluster[j].rhfracSize();
        const auto test_recFracOffset = outPFCluster[j].rhfracOffset();
        const auto test_clidx = outPFRecHitFracs[test_recFracOffset].pfcIdx();
        match = (test_recFracSize == recFracSize) && (test_clidx == j);
        if (match) {
          found_idx = test_clidx;
          nmatch += 1;
        }
      }
    }

    if (nmatch == 0) {
      printf("ERROR: did not find test cluster (alpaka cluster id %u) !\n", clidx);
    } else if (nmatch > 1) {
      printf("ERROR: found multiple clusters (alpaka cluster id %u) !\n", clidx);
    }

    const int recFracOffset4check = outPFCluster[found_idx].rhfracOffset();
    const int recFracSize4check = outPFCluster[found_idx].rhfracSize();

    for (int r = 0; r < recFracSize; r++) {
      const int recHitidx = outHostRecHitsFracsView[recFracOffset + r].pfrhIdx();
      const float frac = outHostRecHitsFracsView[recFracOffset + r].frac();
      bool found = false;

      for (int j = 0; j < recFracSize4check; j++) {
        const int recHitidx4check = outPFRecHitFracs[recFracOffset4check + j].pfrhIdx();
        const float frac4check = outPFRecHitFracs[recFracOffset4check + j].frac();

        if (recHitidx == recHitidx4check && (fabs(frac - frac4check) < 1e-6))
          found = true;
      }

      if (!found) {
        printf("ERROR: could not find rechit idx %u for cluster %u, size %u\n", recHitidx, i, recFracSize4check);
        exit(-1);
      }
    }
  }

  return nerrors;
}

using namespace edm;
using namespace std;

int main(int argc, char **argv) {
  if (argc > 5) {
    std::cerr << "Usage: " << argv[0] << " [<nClusters>] [<nStreams>] [<nIters>] [<threadsPerBlock>]\n";
    return 1;
  }

  // Get the list of devices on the current platform
  auto const &devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
                 "the test will be skipped.\n";
    return EXIT_FAILURE; // Prefer returning over raw exit() in main
  }

  int nStreams = 1;
  int nClusters = multiblock ? 1024 : 512;
  int nIters = nColdIters + 100;

  auto parseArg = [](const char* arg, const std::string& name) {
    try {
      return std::stoi(arg);
    } catch (...) {
      std::cerr << "Error: Invalid integer value provided for " << name << ": '" << arg << "'\n";
      std::exit(EXIT_FAILURE);
    }
  };

  if (argc > 1) {
    nClusters = parseArg(argv[1], "<nClusters>");

    if constexpr (multiblock == false) {
      if (nClusters > 1024) {
        std::cerr << "Abort the test: <nClusters> cannot exceed 1024 for a single-block run.\n";
        return 1;
      }
    }
  }

  if (argc > 2) nStreams = parseArg(argv[2], "<nStreams>");
  if (argc > 3) nIters   = parseArg(argv[3], "<nIters>");

  int threadsPerBlock = multiblock ? 128 : nClusters;

  if (argc > 4) {
    if constexpr (multiblock) {
      threadsPerBlock = parseArg(argv[4], "<threadsPerBlock>");
    } else {
      std::cout << "Warning: <threadsPerBlock> argument ignored for single-block runs (it matches nClusters).\n";
    }
  }

  const int maxHitsPerCluster = 99;
  const int minHitsPerCluster = 1;

  ::reco::PFClusterCollection clusters;
  clusters.reserve(nClusters);

  std::vector<Fraction> rhfracs;
  rhfracs.reserve(nClusters * maxHitsPerCluster);

  PFRecHitCollection hits;
  hits.reserve(nClusters * maxHitsPerCluster);

  std::vector<int> seedIdx(nClusters);

  auto sizes = create(clusters, hits, rhfracs, seedIdx, nClusters, minHitsPerCluster, maxHitsPerCluster);

  const int nHits = sizes.first;
  const int nFracs = sizes.second;

  std::vector<int> cc_roots = {0, 1, 5, 9, 38 , 45, 49, 66, 71, 77, 89, 91, 99 };

  const int cc_num = static_cast<int>(cc_roots.size());

  if (cc_roots.back() > nClusters) {
    std::cerr << "Incorrect cc list." << std::endl;
    exit(-1);
  }

  // run the test on each device
  for (auto const &device : devices) {
    std::vector<Queue> queues;
    queues.reserve(nStreams);

    std::vector<::reco::PFClusterHostCollection> outHostClusters;
    outHostClusters.reserve(nStreams);
    std::vector<::reco::PFRecHitFractionHostCollection> outHostRecHitFracs;
    outHostRecHitFracs.reserve(nStreams);

    std::vector<::reco::PFClusterHostCollection> hostClusters;
    hostClusters.reserve(nStreams);
    std::vector<::reco::PFRecHitHostCollection> hostRecHits;
    hostRecHits.reserve(nStreams);
    std::vector<::reco::PFRecHitFractionHostCollection> hostRecHitFracs;
    hostRecHitFracs.reserve(nStreams);

    std::vector<::reco::PFMultiDepthClusteringCCLabelsHostCollection> hostClusteringVars;
    hostClusteringVars.reserve(nStreams);

    for (int s = 0; s < nStreams; s++) {
      auto queue = Queue(device);

      auto out_host_clusters = ::reco::PFClusterHostCollection(queue, cc_num);

      auto host_clusters = ::reco::PFClusterHostCollection(queue, nClusters);
      auto host_hits = ::reco::PFRecHitHostCollection(queue, nHits);
      auto host_rhfracs = ::reco::PFRecHitFractionHostCollection(queue, nFracs);
      auto host_clustering_vars = ::reco::PFMultiDepthClusteringCCLabelsHostCollection(queue, nClusters);

      auto outHClusters = out_host_clusters.view();
      auto hClusters = host_clusters.view();
      auto hRecHits = host_hits.view();
      auto hClusteringVars = host_clustering_vars.view();

      outHClusters.size() = cc_num;

      hRecHits.size() = nHits;

      hClusters.nTopos() = nClusters;
      hClusters.nSeeds() = nClusters;
      hClusters.nRHFracs() = nFracs;
      hClusters.size() = nClusters;

      hClusteringVars.size() = nClusters;
      hClusteringVars.ncomponents() = 0;

      load(host_clusters, host_hits, host_rhfracs, clusters, hits, rhfracs, seedIdx);

      create_cc_list(host_clustering_vars, cc_roots, nClusters);

      hostClusters.emplace_back(std::move(host_clusters));
      hostRecHits.emplace_back(std::move(host_hits));
      hostRecHitFracs.emplace_back(std::move(host_rhfracs));

      outHostClusters.emplace_back(std::move(out_host_clusters));
      outHostRecHitFracs.emplace_back(::reco::PFRecHitFractionHostCollection(queue, nFracs));
      hostClusteringVars.emplace_back(std::move(host_clustering_vars));

      queues.emplace_back(std::move(queue));
    }

    launch_epilogue_test(queues,
                         outHostClusters,
                         outHostRecHitFracs,
                         hostClusteringVars,
                         hostClusters,
                         hostRecHits,
                         hostRecHitFracs,
                         nStreams,
                         nIters,
                         threadsPerBlock);

    auto nerrs = checkEpilogue(queues[0],
                               outHostClusters[0],
                               outHostRecHitFracs[0],
                               hostClusters[0],
                               hostRecHits[0],
                               hostRecHitFracs[0],
                               hostClusteringVars[0],
                               cc_roots);

    if (nerrs != 0) {
      std::cerr << nerrs << " errors detected, exiting." << std::endl;
      std::exit(-1);
    }
  }

  return 0;
}
