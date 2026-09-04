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
// 21
static constexpr unsigned int nStreams = 1;  //T4:40 RTX:72 L40S:142 4090:128 5080:84

static constexpr unsigned int nIters = 100;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class ConstructLinksTest {
  public:
    void apply(std::vector<Queue>& queues,
               std::vector<reco::PFMultiDepthClusteringCCLabelsDeviceCollection>& mdpfCCLabels,
               const std::vector<reco::PFMultiDepthClusteringVarsDeviceCollection>& mdpfClusteringVars,
               const PFMultiDepthClusterParams* nSigma) const;
  };

  void ConstructLinksTest::apply(std::vector<Queue>& queues,
                                 std::vector<reco::PFMultiDepthClusteringCCLabelsDeviceCollection>& mdpfCCLabels,
                                 const std::vector<reco::PFMultiDepthClusteringVarsDeviceCollection>& mdpfClusteringVars,
                                 const PFMultiDepthClusterParams* nSigma) const {
    uint32_t items = std::is_same_v<Device, alpaka::DevCpu> ? 1 : 128;

    auto n = static_cast<uint32_t>(mdpfClusteringVars[0]->metadata().size());
    uint32_t groups = ::cms::alpakatools::divide_up_by(n, items);

    if (groups < 1) {
      printf("Skip kernel launch...\n");
      return;
    }

    auto workDiv = ::cms::alpakatools::make_workdiv<Acc1D>(groups, items);

    double wall_time = 0.0;
    for (unsigned int i = 0; i < nIters; i++) {
      auto wall_start = std::chrono::high_resolution_clock::now();

      for (unsigned int s = 0; s < nStreams; s++) {
        alpaka::exec<Acc1D>(
            queues[0], workDiv, ConstructLinksKernel{}, mdpfCCLabels[s].view(), mdpfClusteringVars[s].view(), nSigma);
      }

      for (unsigned int s = 0; s < nStreams; s++)
        alpaka::wait(queues[s]);

      auto wall_stop = std::chrono::high_resolution_clock::now();
      //
      auto wall_diff = wall_stop - wall_start;

      wall_time += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6;
    }

    printf("Wall time: %f sec per iter, %f per stream per iter\n", wall_time / nIters, wall_time / (nIters * nStreams));
  }

  void launch_construct_links_test(
      std::vector<Queue>& queues,
      std::vector<::reco::PFMultiDepthClusteringCCLabelsHostCollection>& hostClusteringCCLabels,
      const std::vector<::reco::PFMultiDepthClusteringVarsHostCollection>& hostClusteringVars) {
    ConstructLinksTest construct_links_test{};

    auto params_h = cms::alpakatools::make_host_buffer<PFMultiDepthClusterParams, Platform>();

    params_h->nSigmaEta = nSigmaEta;
    params_h->nSigmaPhi = nSigmaPhi;

    auto params_d = cms::alpakatools::make_device_buffer<PFMultiDepthClusterParams>(queues[0]);

    alpaka::memcpy(queues[0], params_d, params_h);

    auto hClusteringVars = hostClusteringVars[0].view();

    const int nClusters = hClusteringVars.size();

    std::vector<reco::PFMultiDepthClusteringVarsDeviceCollection> devClusteringVars;
    devClusteringVars.reserve(nStreams);

    std::vector<reco::PFMultiDepthClusteringCCLabelsDeviceCollection> devClusteringCCLabels;
    devClusteringVars.reserve(nStreams);

    for (unsigned int s = 0; s < nStreams; s++) {
      //
      auto dev_clustering_vars = reco::PFMultiDepthClusteringVarsDeviceCollection{queues[s], nClusters};
      //
      alpaka::memcpy(queues[s], dev_clustering_vars.buffer(), hostClusteringVars[s].buffer());

      devClusteringVars.emplace_back(std::move(dev_clustering_vars));

      devClusteringCCLabels.emplace_back(
          reco::PFMultiDepthClusteringCCLabelsDeviceCollection(queues[s], nClusters + 1));

      alpaka::wait(queues[s]);
    }

    construct_links_test.apply(queues, devClusteringCCLabels, devClusteringVars, params_d.data());

    alpaka::memcpy(queues[0], hostClusteringCCLabels[0].buffer(), devClusteringCCLabels[0].buffer());

    for (unsigned int s = 0; s < nStreams; s++)
      alpaka::wait(queues[s]);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

void create(::reco::PFClusterCollection& clusters, const int nClusters) {
  std::mt19937 rng(12345);

  // E in [0.5, 120], r in [174, 180], phi in [0, 0.25], z = 0
  std::uniform_real_distribution<double> energies_dist(0.5, 120.0);
  std::uniform_real_distribution<double> r_dist(174.0, 180.0);
  std::uniform_real_distribution<double> phi_dist(0.0, 0.25);
  std::uniform_real_distribution<double> z_dist(0.0, 0.5);

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

    ::reco::PFCluster cluster(layer, energy, x, y, z);
    cluster.setDepth(depth);

    clusters.emplace_back(std::move(cluster));
  }
}

void load(::reco::PFMultiDepthClusteringVarsHostCollection& hostClusteringVars,
          const ::reco::PFClusterCollection& clusters) {
  const int nClusters = clusters.size();
  auto hClusteringVars = hostClusteringVars.view();

  hClusteringVars.size() = nClusters;

  std::mt19937 rng(12355);

  std::uniform_real_distribution<double> epRMS2_dist(0.001, 0.05);

  for (int i = 0; i < nClusters; ++i) {
    const ::reco::PFCluster& cluster = clusters[i];

    hClusteringVars[i].depth() = cluster.depth();
    hClusteringVars[i].energy() = cluster.energy();

    hClusteringVars[i].etaRMS2() = epRMS2_dist(rng);
    hClusteringVars[i].phiRMS2() = epRMS2_dist(rng);

    auto const& crep = cluster.positionREP();

    hClusteringVars[i].eta() = crep.eta();
    hClusteringVars[i].phi() = crep.phi();
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

template <bool verbose = false>
std::vector<ClusterLink> prune(std::vector<ClusterLink>& links, std::vector<bool>& linkedClusters) {
  std::vector<ClusterLink> goodLinks;
  std::vector<bool> mask(links.size(), false);
  if (links.empty())
    return goodLinks;

  if constexpr (verbose)
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

  if constexpr (verbose)
    printf("Total pruned links %u\n", pruned_links);

  if constexpr (verbose)
    printf("Total number of selected links : %lu\n", goodLinks.size());

  return goodLinks;
}

void runPruningTest(const ::reco::PFClusterCollection& clusters,
                    const std::vector<double>& etaRMS2,
                    const std::vector<double>& phiRMS2) {
  double dummyEn = 0.;
  // Time loop
  auto start = std::chrono::steady_clock::now();

  for (unsigned int i = 0; i < nIters; i++) {
    //link
    auto&& links = link(clusters, etaRMS2, phiRMS2);

    std::vector<bool> linked(clusters.size(), false);
    //prune
    auto&& prunedLinks = prune(links, linked);

    dummyEn += prunedLinks[0].energy();
  }

  auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  std::cout << "Legacy function execution time : " << seconds / nIters << " sec per stream per iter." << std::endl;

  std::cout << "Link 0 energy " << dummyEn / nIters << std::endl;
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
  // Check time:
  runPruningTest(clusters, etaRMS2, phiRMS2);

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

  const int nClusters = 1024;

  ::reco::PFClusterCollection clusters;
  clusters.reserve(nClusters);
  create(clusters, nClusters);

  // run the test on each device
  for (auto const& device : devices) {
    std::vector<Queue> queues;
    queues.reserve(nStreams);

    std::vector<::reco::PFMultiDepthClusteringVarsHostCollection> hostClusteringVars;
    hostClusteringVars.reserve(nStreams);

    std::vector<::reco::PFMultiDepthClusteringCCLabelsHostCollection> hostClusteringCCLabels;
    hostClusteringCCLabels.reserve(nStreams);

    for (unsigned int s = 0; s < nStreams; s++) {
      auto queue = Queue(device);

      auto host_clustering_vars = ::reco::PFMultiDepthClusteringVarsHostCollection(queue, nClusters);

      load(host_clustering_vars, clusters);

      hostClusteringVars.emplace_back(std::move(host_clustering_vars));

      hostClusteringCCLabels.emplace_back(::reco::PFMultiDepthClusteringCCLabelsHostCollection(queue, nClusters + 1));

      queues.emplace_back(std::move(queue));
    }

    launch_construct_links_test(queues, hostClusteringCCLabels, hostClusteringVars);

    auto nerrors = checkConstructLinks(clusters, hostClusteringCCLabels[0], hostClusteringVars[0]);

    if (nerrors != 0) {
      std::cerr << nerrors << " errors detected, exiting." << std::endl;
      std::exit(-1);
    }
  }

  return 0;
}
