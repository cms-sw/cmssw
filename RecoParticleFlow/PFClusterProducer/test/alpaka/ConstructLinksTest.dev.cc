#include <random>
#include <vector>
#include <cmath>

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
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthConstructLinks.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringVarsHostCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFMultiDepthClusteringCCLabelsHostCollection.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterParams.h"

constexpr double nSigmaEta = 0.01234;
constexpr double nSigmaPhi = 0.2678;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class ConstructLinksTest {
  public:
    void apply(Queue& queue,
               reco::PFMultiDepthClusteringCCLabelsDeviceCollection& mdpfCCLabels,
               const reco::PFMultiDepthClusteringVarsDeviceCollection& mdpfClusteringVars,
               const PFMultiDepthClusterParams* nSigma) const;
  };

  void ConstructLinksTest::apply(Queue& queue,
                                 reco::PFMultiDepthClusteringCCLabelsDeviceCollection& mdpfCCLabels,
                                 const reco::PFMultiDepthClusteringVarsDeviceCollection& mdpfClusteringVars,
                                 const PFMultiDepthClusterParams* nSigma) const {
#ifdef __CUDACC__
    uint32_t items = 160;
#else
    uint32_t items = 192;
#endif

    auto n = static_cast<uint32_t>(mdpfClusteringVars->metadata().size());
    uint32_t groups = cms::alpakatools::divide_up_by(n, items);

    if (groups < 1) {
      printf("Skip kernel launch...\n");
      return;
    } else {
      printf("Run kernel in %d threads, %d groups.\n", n, groups);
    }

    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);

    alpaka::exec<Acc1D>(queue, workDiv, ConstructLinksKernel{}, mdpfCCLabels.view(), mdpfClusteringVars.view(), nSigma);

    alpaka::wait(queue);
  }

  void launch_construct_links_test(Queue& queue,
                                   ::reco::PFMultiDepthClusteringCCLabelsHostCollection& hostClusteringCCLabels,
                                   const ::reco::PFMultiDepthClusteringVarsHostCollection& hostClusteringVars) {
    ConstructLinksTest construct_links_test{};

    auto hClusteringVars = hostClusteringVars.view();

    const int nClusters = hClusteringVars.size();

    reco::PFMultiDepthClusteringVarsDeviceCollection devClusteringVars{queue, nClusters};

    alpaka::memcpy(queue, devClusteringVars.buffer(), hostClusteringVars.buffer());

    reco::PFMultiDepthClusteringCCLabelsDeviceCollection devClusteringCCLabels{queue, nClusters + 1};

    auto params_h = cms::alpakatools::make_host_buffer<PFMultiDepthClusterParams, Platform>();

    params_h->nSigmaEta = nSigmaEta;
    params_h->nSigmaPhi = nSigmaPhi;

    auto params_d = cms::alpakatools::make_device_buffer<PFMultiDepthClusterParams>(queue);

    alpaka::memcpy(queue, params_d, params_h);

    construct_links_test.apply(queue, devClusteringCCLabels, devClusteringVars, params_d.data());

    alpaka::memcpy(queue, hostClusteringCCLabels.buffer(), devClusteringCCLabels.buffer());

    alpaka::wait(queue);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

void create(::reco::PFClusterCollection& clusters,
            ::reco::PFMultiDepthClusteringVarsHostCollection& hostClusteringVars,
            const int nClusters) {
  auto hClusteringVars = hostClusteringVars.view();

  std::mt19937 rng(12345);

  // E in [0.5, 120], r in [174, 180], phi in [0, 0.25], z = 0
  std::uniform_real_distribution<double> energies_dist(0.5, 120.0);
  std::uniform_real_distribution<double> r_dist(174.0, 180.0);
  std::uniform_real_distribution<double> phi_dist(0.0, 0.25);
  std::uniform_real_distribution<double> z_dist(0.0, 0.5);
  std::uniform_real_distribution<double> epRMS2_dist(0.001, 0.05);

  std::uniform_int_distribution<int> depth_distr(1, 4);

  for (int i = 0; i < nClusters; ++i) {
    const bool is_depth1 = (i % 2 == 0);
    const PFLayer::Layer layer = is_depth1 ? PFLayer::Layer::HCAL_BARREL1 : PFLayer::Layer::HCAL_BARREL2;
    const double depth = depth_distr(rng);

    const double energy = energies_dist(rng);

    const double r = r_dist(rng);
    const double phi = phi_dist(rng);

    const double x = r * std::cos(phi);
    const double y = r * std::sin(phi);
    const double z = z_dist(rng);

    const double etaRMS2_ = epRMS2_dist(rng);
    const double phiRMS2_ = epRMS2_dist(rng);

    hClusteringVars[i].depth() = depth;
    hClusteringVars[i].energy() = energy;

    hClusteringVars[i].etaRMS2() = etaRMS2_;
    hClusteringVars[i].phiRMS2() = phiRMS2_;

    ::reco::PFCluster cluster(layer, energy, x, y, z);
    cluster.setDepth(depth);

    auto const& crep = cluster.positionREP();

    hClusteringVars[i].eta() = crep.eta();
    hClusteringVars[i].phi() = crep.phi();
    clusters.emplace_back(std::move(cluster));
  }
}

class ClusterLink {
public:
  ClusterLink(unsigned int i, unsigned int j, double DR, int DZ, double energy) {
    from_ = i;
    to_ = j;
    linkDR_ = DR;
    linkDZ_ = DZ;
    linkE_ = energy;
  }

  ~ClusterLink() = default;

  unsigned int from() const { return from_; }
  unsigned int to() const { return to_; }
  double dR() const { return linkDR_; }
  int dZ() const { return linkDZ_; }
  double energy() const { return linkE_; }

private:
  unsigned int from_;
  unsigned int to_;
  double linkDR_;
  int linkDZ_;
  double linkE_;
};

std::vector<ClusterLink> link(const ::reco::PFClusterCollection& clusters,
                              const std::vector<double>& etaRMS2,
                              const std::vector<double>& phiRMS2) {
  std::vector<ClusterLink> links;
  //loop on all pairs
  for (unsigned int i = 0; i < clusters.size(); ++i)
    for (unsigned int j = 0; j < clusters.size(); ++j) {
      if (i == j)
        continue;

      const ::reco::PFCluster& cluster1 = clusters[i];
      const ::reco::PFCluster& cluster2 = clusters[j];

      // PFCluster depth stored as double but HCAL layer clusters have integral depths only
      auto dz = (static_cast<int>(cluster2.depth()) - static_cast<int>(cluster1.depth()));

      //Do not link at the same layer and only link inside out!
      if (dz <= 0)
        continue;
      //printf("dZ check %d\n", std::abs(dz));
      auto const& crep1 = cluster1.positionREP();
      auto const& crep2 = cluster2.positionREP();

      auto deta = crep1.eta() - crep2.eta();
      deta = deta * deta / (etaRMS2[i] + etaRMS2[j]);
      auto dphi = deltaPhi(crep1.phi(), crep2.phi());
      dphi = dphi * dphi / (phiRMS2[i] + phiRMS2[j]);
      if ((deta < nSigmaEta) & (dphi < nSigmaPhi)) {
        links.push_back(ClusterLink(i, j, deta + dphi, std::abs(dz), cluster1.energy() + cluster2.energy()));
      }
    }

  return links;
}

std::vector<ClusterLink> prune(std::vector<ClusterLink>& links, std::vector<bool>& linkedClusters) {
  std::vector<ClusterLink> goodLinks;
  std::vector<bool> mask(links.size(), false);
  if (links.empty())
    return goodLinks;

  printf("Total number of links : %lu\n", links.size());

  for (unsigned int i = 0; i < links.size() - 1; ++i) {
    if (mask[i])
      continue;
    for (unsigned int j = i + 1; j < links.size(); ++j) {
      if (mask[j])
        continue;

      const ClusterLink& link1 = links[i];
      const ClusterLink& link2 = links[j];

      if (link1.to() == link2.to()) {  //found two links going to the same spot,kill one
        //first prefer nearby layers
        if (link1.dZ() < link2.dZ()) {
          mask[j] = true;
        } else if (link1.dZ() > link2.dZ()) {
          mask[i] = true;
        } else {  //if same layer-pick based on transverse compatibility
          if (link1.dR() < link2.dR()) {
            mask[j] = true;
          } else if (link1.dR() > link2.dR()) {
            mask[i] = true;
          } else {
            //same distance as well -> can happen !!!!! Pick the highest SUME
            if (link1.energy() < link2.energy())
              mask[i] = true;
            else
              mask[j] = true;
          }
        }
      }
    }
  }

  unsigned int pruned_links = 0;

  for (unsigned int i = 0; i < links.size(); ++i) {
    if (mask[i]) {
      pruned_links += 1;
      continue;
    }
    goodLinks.push_back(links[i]);

    linkedClusters[links[i].from()] = true;
    linkedClusters[links[i].to()] = true;
  }

  printf("Total pruned links %u\n", pruned_links);

  printf("Total number of selected links : %lu\n", goodLinks.size());

  return goodLinks;
}

int checkConstructLinks(const ::reco::PFClusterCollection& clusters,
                        const ::reco::PFMultiDepthClusteringCCLabelsHostCollection& hostClusteringCCLabels,
                        const ::reco::PFMultiDepthClusteringVarsHostCollection& hostClusteringVars) {
  const int nClusters = clusters.size();

  auto hClusteringVars = hostClusteringVars.view();

  std::vector<double> etaRMS2(nClusters, 0.0);
  std::vector<double> phiRMS2(nClusters, 0.0);

  for (int i = 0; i < nClusters; ++i) {
    etaRMS2[i] = hClusteringVars[i].etaRMS2();
    phiRMS2[i] = hClusteringVars[i].phiRMS2();
  }
  //link
  auto&& links = link(clusters, etaRMS2, phiRMS2);

  std::vector<bool> linked(clusters.size(), false);
  //prune
  auto&& prunedLinks = prune(links, linked);

  auto hClusteringCCLabels = hostClusteringCCLabels.view();

  int nerrors = 0;
  for (int i = 0; i < nClusters; i++) {
    if (i == hClusteringCCLabels[i].mdpf_topoId())
      printf("Check self connection: vertex id %d is linked to itself (linked to others : %s)\n",
             hClusteringCCLabels[i].mdpf_topoId(),
             linked[i] ? "Y" : "N");
    if (i != hClusteringCCLabels[i].mdpf_topoId() && (linked[i] == false)) {
      printf("Error: vertex id %d must not be linked to %d.\n", hClusteringCCLabels[i].mdpf_topoId(), i);
      nerrors += 1;
    }
  }

  for (auto& link_ : prunedLinks) {
    const int from_link_id = link_.from();
    const int to_link_id = link_.to();
    //
    if (from_link_id != hClusteringCCLabels[to_link_id].mdpf_topoId()) {
      printf("Error: src ver %d linked to dest ver %d, while on Alpaka device dest ver %d linked to src ver id %d...\n",
             from_link_id,
             to_link_id,
             to_link_id,
             hClusteringCCLabels[to_link_id].mdpf_topoId());
      nerrors += 1;
    }
  }
  return nerrors;
}

using namespace edm;
using namespace std;

int main() {
  // get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  const int nClusters = 145;

  // run the test on each device
  for (auto const& device : devices) {
    auto queue = Queue(device);

    ::reco::PFClusterCollection clusters;
    clusters.reserve(nClusters);

    ::reco::PFMultiDepthClusteringVarsHostCollection hostClusteringVars{queue, nClusters};

    ::reco::PFMultiDepthClusteringCCLabelsHostCollection hostClusteringCCLabels{queue, nClusters + 1};

    auto hClusteringVars = hostClusteringVars.view();
    hClusteringVars.size() = nClusters;

    create(clusters, hostClusteringVars, nClusters);

    launch_construct_links_test(queue, hostClusteringCCLabels, hostClusteringVars);

    auto nerrors = checkConstructLinks(clusters, hostClusteringCCLabels, hostClusteringVars);

    if (nerrors != 0) {
      std::cerr << nerrors << " errors detected, exiting." << std::endl;
      std::exit(-1);
    }
  }

  return 0;
}
