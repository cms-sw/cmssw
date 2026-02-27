#include <random>
#include <vector>
#include <cmath>
#include <algorithm>

#include <iostream>
#include <cstdlib>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <alpaka/alpaka.hpp>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizerHelper.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthShowerShape.h"

#include "DataFormats/ParticleFlowReco/interface/PFClusterHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFClusterDeviceCollection.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitFractionDeviceCollection.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometryMayOwnPtr.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringVarsHostCollection.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#ifdef SHOWER_SHAPE_COOPERATIVE
static constexpr bool cooperative = true;
#else
static constexpr bool cooperative = false;
#endif

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

  CaloCellGeometry::CornersMgr s_cornersMgr(65536,
                                            CaloCellGeometry::k_cornerSize);  //k_cornerSize = 8;

  CaloCellGeometry::ParMgr s_parMgr(65536, /*subSize=*/BoxCell::kNPar);

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
  public:
    void apply(Queue& queue,
               reco::PFMultiDepthClusteringVarsDeviceCollection& mdpfClusteringVars,
               const reco::PFClusterDeviceCollection& pfClusters,
               const reco::PFRecHitFractionDeviceCollection& pfRecHitFracs,
               const reco::PFRecHitDeviceCollection& pfRecHit) const;
  };

  void ShowerShapeTest::apply(Queue& queue,
                              reco::PFMultiDepthClusteringVarsDeviceCollection& mdpfClusteringVars,
                              const reco::PFClusterDeviceCollection& pfClusters,
                              const reco::PFRecHitFractionDeviceCollection& pfRecHitFracs,
                              const reco::PFRecHitDeviceCollection& pfRecHit) const {
    uint32_t items = 160;

    auto n = static_cast<uint32_t>(pfClusters->metadata().size());
    uint32_t groups = ::cms::alpakatools::divide_up_by(n, items);

    if (groups < 1) {
      printf("Skip kernel launch...\n");
      return;
    }

    auto workDiv = ::cms::alpakatools::make_workdiv<Acc1D>(groups, items);

    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        ShowerShapeKernel<cooperative>{},
                        mdpfClusteringVars.view(),
                        pfClusters.view(),
                        pfRecHitFracs.view(),
                        pfRecHit.view(),
                        thrsh);

    alpaka::wait(queue);
  }

  void launch_shower_shape_test(Queue& queue,
                                ::reco::PFMultiDepthClusteringVarsHostCollection& hostClusteringVars,
                                const ::reco::PFClusterHostCollection& hostClusters,
                                const ::reco::PFRecHitHostCollection& hostRecHits,
                                const ::reco::PFRecHitFractionHostCollection& hostRecHitFracs) {
    ShowerShapeTest shower_shape_test{};

    auto hClusters = hostClusters.view();
    auto hRecHits = hostRecHits.view();

    const int nClusters = hClusters.size();
    const int nHits = hRecHits.size();
    const int nFracs = hClusters.nRHFracs();

    reco::PFClusterDeviceCollection devClusters{queue, nClusters};
    reco::PFRecHitDeviceCollection devRecHits{queue, nHits};
    reco::PFRecHitFractionDeviceCollection devRecHitFracs{queue, nFracs};

    alpaka::memcpy(queue, devClusters.buffer(), hostClusters.buffer());
    alpaka::memcpy(queue, devRecHits.buffer(), hostRecHits.buffer());
    alpaka::memcpy(queue, devRecHitFracs.buffer(), hostRecHitFracs.buffer());

    reco::PFMultiDepthClusteringVarsDeviceCollection devClusteringVars{queue, nClusters};

    shower_shape_test.apply(queue, devClusteringVars, devClusters, devRecHitFracs, devRecHits);

    alpaka::memcpy(queue, hostClusteringVars.buffer(), devClusteringVars.buffer());
    alpaka::wait(queue);
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
                           const float shareProbability = 0.05f) {
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

    const float leftover = std::max(0.f, 1.f - ownerFrac);
    if (leftover <= 0.f)
      continue;

    // give the neighbor between 20% and 80% of the leftover
    const float share_frac = leftover * (0.2f + 0.6f * uni_distr(rng));

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

int checkShowerShapes(const ::reco::PFClusterCollection& clusters,
                      const ::reco::PFRecHitCollection& recHits,
                      const ::reco::PFMultiDepthClusteringVarsHostCollection& hostClusteringVars) {
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

  int nerrors = 0;

  auto hClusteringVars = hostClusteringVars.view();
  double tol = 5e-6;

  for (int i = 0; i < nClusters; i++) {
    const auto x = std::abs(hClusteringVars[i].etaRMS2() - etaRMS2[i]) / etaRMS2[i];
    if (x > tol) {
      printf("Result for cluster id %d : etaRMS2 %f (%f), %f\n", i, hClusteringVars[i].etaRMS2(), etaRMS2[i], x);
      nerrors += 1;
    }
    const auto y = std::abs(hClusteringVars[i].phiRMS2() - phiRMS2[i]) / phiRMS2[i];
    if (y > tol) {
      printf("Result for cluster id %d : phiRMS2 %f (%f), %f\n", i, hClusteringVars[i].phiRMS2(), phiRMS2[i], y);
      nerrors += 1;
    }
  }
  return nerrors;
}

using namespace edm;
using namespace std;

int main() {
  // get the list of devices on the current platform
  auto const& devices = ::cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  const int nClusters = 145;
  const int maxHitsPerCluster = 67;
  const int minHitsPerCluster = 23;

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
    auto queue = Queue(device);

    ::reco::PFClusterHostCollection hostClusters{queue, nClusters};
    ::reco::PFRecHitHostCollection hostRecHits{queue, nHits};
    ::reco::PFRecHitFractionHostCollection hostRecHitFracs{queue, nFracs};

    auto hClusters = hostClusters.view();
    auto hRecHits = hostRecHits.view();

    hRecHits.size() = nHits;

    hClusters.nTopos() = nClusters;
    hClusters.nSeeds() = nClusters;
    hClusters.nRHFracs() = nFracs;
    hClusters.size() = nClusters;

    load(hostClusters, hostRecHits, hostRecHitFracs, clusters, hits, rhfracs, seedIdx);

    ::reco::PFMultiDepthClusteringVarsHostCollection hostClusteringVars{queue, nClusters};

    launch_shower_shape_test(queue, hostClusteringVars, hostClusters, hostRecHits, hostRecHitFracs);

    auto nerrors = checkShowerShapes(clusters, hits, hostClusteringVars);

    if (nerrors != 0) {
      std::cerr << nerrors << " errors detected, exiting." << std::endl;
      std::exit(-1);
    }
  }

  return 0;
}
