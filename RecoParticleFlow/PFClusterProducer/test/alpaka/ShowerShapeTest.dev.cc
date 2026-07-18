#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <Eigen/Core>
#include <Eigen/Dense>
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
#include "FWCore/Utilities/interface/stringize.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometryMayOwnPtr.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizerHelper.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringVarsHostCollection.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthShowerShape.h"

#ifdef SHOWER_SHAPE_COOPERATIVE
static constexpr bool cooperative = true;
#else
static constexpr bool cooperative = false;
#endif
// 21
//static constexpr unsigned int nStreams = 32;  //T4:40 RTX:72 L40S:142 4090:128 5080:84

static constexpr int nColdIters = 10;

using PFRecHitsNeighbours = Eigen::Matrix<int32_t, 8, 1>;

using namespace reco;

static bool verbose = true;
constexpr float thrsh = 0.1f;

// Simple axis-aligned test cell:
namespace {
  class BoxCell final : public CaloCellGeometry {
  public:
    enum ParIdx { kDx = 0, kDy = 1, kDz = 2, kNPar = 3 };

    BoxCell(CornersMgr* cMgr, GlobalPoint const& center, CCGFloat const* par) : CaloCellGeometry(center, cMgr, par) {
      initSpan();
    }

    void vocalCorners(Pt3DVec& out, CCGFloat const*, Pt3D& ref) const final {
      out.clear();
      out.reserve(k_cornerSize);
      auto const& cv = getCorners();
      for (unsigned i = 0; i < k_cornerSize; ++i)
        out.emplace_back(cv[i].x(), cv[i].y(), cv[i].z());
      auto const& P = getPosition();
      ref = Pt3D(P.x(), P.y(), P.z());
    }

    // compute 8 corners from (center, dx,dy,dz) stored in param()
    void initCorners(CornersVec& c) final {
      auto const* p = param();  // points to 3 floats: dx,dy,dz
      auto const dx = p[kDx], dy = p[kDy], dz = p[kDz];
      auto const& C = getPosition();

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

  CaloCellGeometry::CornersMgr s_cornersMgr(4 * 1048576,
                                            CaloCellGeometry::k_cornerSize);  //k_cornerSize = 8;

  CaloCellGeometry::ParMgr s_parMgr(4 * 1048576, /*subSize=*/BoxCell::kNPar);

  CaloCellGeometry::ParVecVec s_parBlocks;

  const CaloCellGeometry::CCGFloat* s_boxParPtr = [] {
    using CCF = CaloCellGeometry::CCGFloat;
    std::vector<CCF> pars = {1.f, 1.f, 1.f};
    return CaloCellGeometry::getParmPtr(pars, &s_parMgr, s_parBlocks);
  }();

  CaloCellGeometryMayOwnPtr makeBoxCellGeo(float x, float y, float z) {
    auto base = std::make_unique<BoxCell>(&s_cornersMgr, GlobalPoint(x, y, z), s_boxParPtr);
    return CaloCellGeometryMayOwnPtr(std::move(base));
  }
}  // namespace

inline HcalDetId makeDetId(int ieta, int iphi, int depth) {
  iphi = std::clamp(iphi, 1, 72);
  ieta = std::clamp(ieta, 1, 16);
  depth = (depth <= 1) ? 1 : 2;

  return HcalDetId(HcalBarrel, ieta, iphi, depth);
}

inline reco::PFRecHit makePFRecHit(
    PFLayer::Layer layer, const HcalDetId& detId, float energy, float x, float y, float z, uint32_t flags = 0) {
  // half-sizes for a simple cell; tweak as you like
  return reco::PFRecHit{makeBoxCellGeo(x, y, z), detId.rawId(), layer, energy, flags};
}

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class ShowerShapeTest {
    const int nStreams;
    const int nIters;
    const int threads;

  public:
    ShowerShapeTest(const int nStreams, const int nIters, const int threads)
        : nStreams(nStreams), nIters(nIters), threads(threads) {}

    void apply(std::vector<Queue>& queues,
               std::vector<reco::PFMultiDepthClusteringVarsDeviceCollection>& mdpfClusteringVars,
               const std::vector<reco::PFClusterDeviceCollection>& pfClusters,
               const std::vector<reco::PFRecHitFractionDeviceCollection>& pfRecHitFracs,
               const std::vector<reco::PFRecHitDeviceCollection>& pfRecHit) const;
  };

  void ShowerShapeTest::apply(std::vector<Queue>& queues,
                              std::vector<reco::PFMultiDepthClusteringVarsDeviceCollection>& mdpfClusteringVars,
                              const std::vector<reco::PFClusterDeviceCollection>& pfClusters,
                              const std::vector<reco::PFRecHitFractionDeviceCollection>& pfRecHitFracs,
                              const std::vector<reco::PFRecHitDeviceCollection>& pfRecHit) const {
    uint32_t items = std::is_same_v<Device, alpaka::DevCpu> ? 1 : threads;

    auto n = static_cast<uint32_t>(pfClusters[0]->metadata().size());
    uint32_t groups = ::cms::alpakatools::divide_up_by(n, items);

    if (groups < 1) {
      printf("Skip kernel launch...\n");
      return;
    }

    auto workDiv = ::cms::alpakatools::make_workdiv<Acc1D>(groups, items);

    double wall_time = 0.0;
    for (int i = 0; i < nIters; i++) {
      auto wall_start = std::chrono::high_resolution_clock::now();

      for (int s = 0; s < nStreams; s++) {
        alpaka::exec<Acc1D>(queues[s],
                            workDiv,
                            ShowerShapeKernel<cooperative>{},
                            mdpfClusteringVars[s].view(),
                            pfClusters[s].view(),
                            pfRecHitFracs[s].view(),
                            pfRecHit[s].view(),
                            thrsh);
      }

      for (int s = 0; s < nStreams; s++)
        alpaka::wait(queues[s]);

      auto wall_stop = std::chrono::high_resolution_clock::now();
      //
      auto wall_diff = wall_stop - wall_start;

      if (i > nColdIters)
        wall_time +=
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
    }

    printf("Wall time: %f sec per iter, %f per stream per iter\n",
           wall_time / (nIters - nColdIters),
           wall_time / ((nIters - nColdIters) * nStreams));
  }

  void launch_shower_shape_test(std::vector<Queue>& queues,
                                std::vector<::reco::PFMultiDepthClusteringVarsHostCollection>& hostClusteringVars,
                                const std::vector<::reco::PFClusterHostCollection>& hostClusters,
                                const std::vector<::reco::PFRecHitHostCollection>& hostRecHits,
                                const std::vector<::reco::PFRecHitFractionHostCollection>& hostRecHitFracs,
                                const int nStreams,
                                const int nIters,
                                const int threadsPerBlock) {
    ShowerShapeTest shower_shape_test(nStreams, nIters, threadsPerBlock);

    const auto hClusters = hostClusters[0].const_view();
    const auto hRecHits = hostRecHits[0].const_view();

    const int nClusters = hClusters.size();
    const int nHits = hRecHits.size();
    const int nFracs = hClusters.nRHFracs();

    std::vector<reco::PFClusterDeviceCollection> devClusters;
    devClusters.reserve(nStreams);
    std::vector<reco::PFRecHitDeviceCollection> devRecHits;
    devRecHits.reserve(nStreams);
    std::vector<reco::PFRecHitFractionDeviceCollection> devRecHitFracs;
    devRecHitFracs.reserve(nStreams);

    std::vector<reco::PFMultiDepthClusteringVarsDeviceCollection> devClusteringVars;
    devClusteringVars.reserve(nStreams);

    for (int s = 0; s < nStreams; s++) {
      //
      auto dev_clusters = reco::PFClusterDeviceCollection{queues[s], nClusters};
      auto dev_hits = reco::PFRecHitDeviceCollection{queues[s], nHits};
      auto dev_rhfrac = reco::PFRecHitFractionDeviceCollection{queues[s], nFracs};
      //
      alpaka::memcpy(queues[s], dev_clusters.buffer(), hostClusters[s].buffer());
      alpaka::memcpy(queues[s], dev_hits.buffer(), hostRecHits[s].buffer());
      alpaka::memcpy(queues[s], dev_rhfrac.buffer(), hostRecHitFracs[s].buffer());

      devClusters.emplace_back(std::move(dev_clusters));
      devRecHits.emplace_back(std::move(dev_hits));
      devRecHitFracs.emplace_back(std::move(dev_rhfrac));

      devClusteringVars.emplace_back(reco::PFMultiDepthClusteringVarsDeviceCollection(queues[s], nClusters));
    }

    shower_shape_test.apply(queues, devClusteringVars, devClusters, devRecHitFracs, devRecHits);

    alpaka::memcpy(queues[0], hostClusteringVars[0].buffer(), devClusteringVars[0].buffer());
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

std::pair<int, int> create(::reco::PFClusterCollection& clusters,
                           ::reco::PFRecHitCollection& hits,
                           std::vector<Fraction>& rhfracs,
                           std::vector<int>& seedIdx,
                           const int nClusters,
                           const int minHitsPerCluster = 2,
                           const int maxHitsPerCluster = 10,
                           const float shareProbability = 0.67f) {
  std::mt19937 rng(12345);

  // E in [0.5, 120], r in [174, 180], phi in [0, 0.25], z = 0
  std::uniform_real_distribution<float> energies_dist(0.5f, 120.0f);
  std::uniform_real_distribution<float> r_dist(174.0f, 180.0f);
  std::uniform_real_distribution<float> phi_dist(0.f, 0.25f);
  std::uniform_real_distribution<float> z_dist(0.f, 0.5f);

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
    const float depth = is_depth1 ? 1.f : 2.f;

    const float energy = energies_dist(rng);

    const float r = r_dist(rng);
    const float phi = phi_dist(rng);

    const float x = r * std::cos(phi);
    const float y = r * std::sin(phi);
    const float z = z_dist(rng);

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

    const float phc = std::atan2(y, x);
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
      constexpr float extra_scale = 10000.0f;  //for testing only
      auto hit = makePFRecHit(layer, detId, extra_scale * eH, hx, hy, hz);
      hitOwners.push_back(i);

      // primary fraction entry: this hit contributes to its owner cluster:
      const float frac_value = (cluster.energy() > 0.0) ? (eH / float(cluster.energy())) : 0.f;
      rhfracs.push_back(Fraction{hIdx, i, frac_value});

      float current_energy = frac_value * hit.energy();

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
  // update cluster collection with rechits:
  for (int j = 0; j < nFracs; j++) {
    ::reco::PFRecHitRef refHit(&hits, rhfracs[j].pfrhIdx);
    clusters[rhfracs[j].pfcIdx].addRecHitFraction(
        ::reco::PFRecHitFraction(refHit, static_cast<double>(rhfracs[j].frac)));
  }

  if (verbose)
    printf("Generated cluster/rechit collections with %d hits and %d rechit fractions.\n", nHits, nFracs);

  return std::make_pair(nHits, nFracs);
}

void load(::reco::PFClusterHostCollection& hostClusters,
          ::reco::PFRecHitHostCollection& hostRecHits,
          ::reco::PFRecHitFractionHostCollection& hostRecHitFracs,
          const ::reco::PFClusterCollection& clusters,
          const ::reco::PFRecHitCollection& hits,
          const std::vector<Fraction>& rhfracs,
          const std::vector<int>& seedIdx) {
  auto hClusters = hostClusters.view();
  auto hRecHits = hostRecHits.view();
  auto hRecHitFracs = hostRecHitFracs.view();

  const int nClusters = hClusters.size();
  const int nHits = hRecHits.size();
  const int nFracs = hClusters.nRHFracs();

  for (int i = 0; i < nHits; ++i) {
    const ::reco::PFRecHit& rhit = hits[i];

    const auto& rhpos = rhit.position();

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
    const ::reco::PFCluster& cluster = clusters[i];

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

    const auto& cpos = cluster.position();

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
  }
}

template <std::floating_point compute_t, std::floating_point reduce_t>
inline void calculateShowerShapes(const ::reco::PFClusterCollection& clusters,
                                  std::vector<reduce_t>& etaRMS2,
                                  std::vector<reduce_t>& phiRMS2) {
  const unsigned int nClusters = clusters.size();

#pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < nClusters; ++i) {
    const ::reco::PFCluster& cluster = clusters[i];

    reduce_t etaSum = 0.0;
    reduce_t phiSum = 0.0;

    auto const& crep = cluster.positionREP();
    auto const& fractions = cluster.recHitFractions();

    const unsigned int nFractions = fractions.size();

#pragma omp simd reduction(+ : etaSum, phiSum)
    for (unsigned int j = 0; j < nFractions; ++j) {
      const auto& frac = fractions[j];

      auto const& h = *frac.recHitRef();
      auto const& rep = h.positionREP();

      const compute_t frcxenergy = static_cast<compute_t>(frac.fraction()) * static_cast<compute_t>(h.energy());

      const compute_t rep_eta = rep.eta();
      const compute_t crep_eta = crep.eta();

      etaSum += (frcxenergy)*std::abs(rep_eta - crep_eta);

      const compute_t rep_phi = rep.phi();
      const compute_t crep_phi = crep.phi();

      phiSum += (frcxenergy)*std::abs(deltaPhi(rep_phi, crep_phi));
    }

    const compute_t inv_energy = 1.f / cluster.energy();
    etaRMS2[i] = std::max(etaSum * inv_energy, static_cast<reduce_t>(thrsh));
    etaRMS2[i] *= etaRMS2[i];
    phiRMS2[i] = std::max(phiSum * inv_energy, static_cast<reduce_t>(thrsh));
    phiRMS2[i] *= phiRMS2[i];
  }
}

void runShowerShapesTest(const ::reco::PFClusterCollection& clusters, const int nIters) {
  using reduce_t = double;
  using compute_t = double;

  std::vector<reduce_t> etaRMS2(clusters.size(), 0);
  std::vector<reduce_t> phiRMS2(clusters.size(), 0);

  // Time loop
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < nIters; i++) {
    //calculate cluster shapes
    calculateShowerShapes<compute_t, reduce_t>(clusters, etaRMS2, phiRMS2);
  }

  auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  std::cout << "Legacy function execution time : " << seconds / nIters << " sec per stream per iter." << std::endl;
}

std::vector<std::pair<int, double>> checkShowerShapes(
    const ::reco::PFClusterCollection& clusters,
    const ::reco::PFRecHitCollection& recHits,
    const ::reco::PFMultiDepthClusteringVarsHostCollection& hostClusteringVars) {
  runShowerShapesTest(clusters, 1000);

  const int nClusters = clusters.size();

  std::vector<double> etaRMS2(nClusters, 0.0);
  std::vector<double> phiRMS2(nClusters, 0.0);

  for (int i = 0; i < nClusters; ++i) {
    const int cluster_idx = i;

    const ::reco::PFCluster& cluster = clusters[cluster_idx];

    double etaSum = 0.0;
    double phiSum = 0.0;

    auto const& crep = cluster.positionREP();
    for (const auto& frac : cluster.recHitFractions()) {
      auto const& h = *frac.recHitRef();
      auto const& rep = h.positionREP();
      etaSum += (frac.fraction() * h.energy()) * std::abs(rep.eta() - crep.eta());
      phiSum += (frac.fraction() * h.energy()) * std::abs(deltaPhi(rep.phi(), crep.phi()));
    }
    //protection for single line : assign ~ tower
    etaRMS2[i] = std::max(etaSum / cluster.energy(), static_cast<double>(thrsh));
    etaRMS2[i] *= etaRMS2[i];
    phiRMS2[i] = std::max(phiSum / cluster.energy(), static_cast<double>(thrsh));
    phiRMS2[i] *= phiRMS2[i];
  }

  auto hClusteringVars = hostClusteringVars.view();

  std::vector<std::pair<int, double>> errors_record;
  errors_record.reserve(6);

  double tol = 5e-6;
  constexpr double tol_step = 1e-1;

  while (tol <= 5e-1) {
    int errs = 0;
    for (int i = 0; i < nClusters; i++) {
      const auto x = std::abs(hClusteringVars[i].etaRMS2() - etaRMS2[i]) / etaRMS2[i];
      if (x > tol) {
        printf("Result for cluster id %d\t and tol %f \t: etaRMS2 %f (%f), %f\n",
               i,
               tol,
               hClusteringVars[i].etaRMS2(),
               etaRMS2[i],
               x);
        errs += 1;
      }
      const auto y = std::abs(hClusteringVars[i].phiRMS2() - phiRMS2[i]) / phiRMS2[i];
      if (y > tol) {
        printf("Result for cluster id %d\t and tol %f \t: phiRMS2 %f (%f), %f\n",
               i,
               tol,
               hClusteringVars[i].phiRMS2(),
               phiRMS2[i],
               y);
        errs += 1;
      }
    }
    errors_record.push_back(std::make_pair(errs, tol));
    tol /= tol_step;
  }
  return errors_record;
}

using namespace edm;
using namespace std;

int main(int argc, char** argv) {
  if (argc > 5) {
    std::cerr << "Usage: " << argv[0] << " <nClusters> <nStreams> <nIters> <threadsPerBlock>\n";
    return 1;
  }
  // get the list of devices on the current platform
  auto const& devices = ::cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  int nStreams = 1;
  int nClusters =  1024;
  int nIters =  nColdIters + 10;

  if (argc > 1) {
    nClusters = std::stoi(argv[1]);
  }

  if (argc > 2)
    nStreams = std::stoi(argv[2]);

  if (argc > 3)
    nIters = std::stoi(argv[3]);

  int threadsPerBlock = nClusters > 1024 ? 128 : nClusters;

  if (argc > 4) {
    threadsPerBlock = std::stoi(argv[4]);
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

  // run the test on each device
  for (auto const& device : devices) {
    std::vector<Queue> queues;
    queues.reserve(nStreams);

    std::vector<::reco::PFClusterHostCollection> hostClusters;
    hostClusters.reserve(nStreams);
    std::vector<::reco::PFRecHitHostCollection> hostRecHits;
    hostRecHits.reserve(nStreams);
    std::vector<::reco::PFRecHitFractionHostCollection> hostRecHitFracs;
    hostRecHitFracs.reserve(nStreams);

    std::vector<::reco::PFMultiDepthClusteringVarsHostCollection> hostClusteringVars;
    hostClusteringVars.reserve(nStreams);

    for (int s = 0; s < nStreams; s++) {
      auto queue = Queue(device);

      auto host_clusters = ::reco::PFClusterHostCollection(queue, nClusters);
      auto host_hits = ::reco::PFRecHitHostCollection(queue, nHits);
      auto host_rhfracs = ::reco::PFRecHitFractionHostCollection(queue, nFracs);

      auto hClusters = host_clusters.view();
      auto hRecHits = host_hits.view();

      hRecHits.size() = nHits;

      hClusters.nTopos() = nClusters;
      hClusters.nSeeds() = nClusters;
      hClusters.nRHFracs() = nFracs;
      hClusters.size() = nClusters;

      load(host_clusters, host_hits, host_rhfracs, clusters, hits, rhfracs, seedIdx);

      hostClusters.emplace_back(std::move(host_clusters));
      hostRecHits.emplace_back(std::move(host_hits));
      hostRecHitFracs.emplace_back(std::move(host_rhfracs));

      hostClusteringVars.emplace_back(::reco::PFMultiDepthClusteringVarsHostCollection(queue, nClusters));

      queues.emplace_back(std::move(queue));
    }

    launch_shower_shape_test(
        queues, hostClusteringVars, hostClusters, hostRecHits, hostRecHitFracs, nStreams, nIters, threadsPerBlock);

    auto errors_record = checkShowerShapes(clusters, hits, hostClusteringVars[0]);

    int nerrors = 0;

    for (auto& e : errors_record) {
      nerrors += e.first;
      std::cout << "Tolerance : " << e.second << "\t, errors : " << e.first << "\t" << std::endl;
    }
    if (nerrors != 0) {
      std::cerr << nerrors << " errors detected, done." << std::endl;
    }
  }

  return 0;
}
