#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/corrector.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/ParametricResolution.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/CaloClusterer.h"

namespace l1tpf {
  class PFClusterProducerFromL1EGClusters : public edm::stream::EDProducer<> {
  public:
    explicit PFClusterProducerFromL1EGClusters(const edm::ParameterSet &);
    ~PFClusterProducerFromL1EGClusters() override {}

  private:
    edm::EDGetTokenT<BXVector<l1t::EGamma>> src_;
    double etCut_;
    std::vector<double> const etaBounds_;
    std::vector<double> const phiBounds_;
    std::vector<unsigned int> const maxClustersEtaPhi_;
    l1tpf::corrector corrector_;
    l1tpf::ParametricResolution resol_;

    void produce(edm::Event &, const edm::EventSetup &) override;

  };  // class
}  // namespace l1tpf

l1tpf::PFClusterProducerFromL1EGClusters::PFClusterProducerFromL1EGClusters(const edm::ParameterSet &iConfig)
    : src_(consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<edm::InputTag>("src"))),
      etCut_(iConfig.getParameter<double>("etMin")),
      etaBounds_(iConfig.getParameter<std::vector<double>>("etaBounds")),
      phiBounds_(iConfig.getParameter<std::vector<double>>("phiBounds")),
      maxClustersEtaPhi_(iConfig.getParameter<std::vector<unsigned int>>("maxClustersEtaPhi")),
      corrector_(iConfig.getParameter<std::string>("corrector"), -1),
      resol_(iConfig.getParameter<edm::ParameterSet>("resol")) {
  produces<l1t::PFClusterCollection>("all");
  produces<l1t::PFClusterCollection>("selected");
  if ((etaBounds_.size() - 1) * (phiBounds_.size() - 1) != maxClustersEtaPhi_.size()) {
    throw cms::Exception("Configuration")
        << "Size mismatch between eta/phi bounds and max clusters: " << (etaBounds_.size() - 1) << " x "
        << (phiBounds_.size() - 1) << " != " << maxClustersEtaPhi_.size() << "\n";
  }
  if (!std::is_sorted(etaBounds_.begin(), etaBounds_.end())) {
    throw cms::Exception("Configuration") << "etaBounds is not sorted\n";
  }
  if (!std::is_sorted(phiBounds_.begin(), phiBounds_.end())) {
    throw cms::Exception("Configuration") << "phiBounds is not sorted\n";
  }
}

void l1tpf::PFClusterProducerFromL1EGClusters::produce(edm::Event &iEvent, const edm::EventSetup &) {
  std::unique_ptr<l1t::PFClusterCollection> out(new l1t::PFClusterCollection());
  std::unique_ptr<l1t::PFClusterCollection> out_sel(new l1t::PFClusterCollection());
  edm::Handle<BXVector<l1t::EGamma>> clusters;
  iEvent.getByToken(src_, clusters);

  l1tpf_calo::GridSelector selector = l1tpf_calo::GridSelector(etaBounds_, phiBounds_, maxClustersEtaPhi_);

  unsigned int index = 0;
  for (auto it = clusters->begin(), ed = clusters->end(); it != ed; ++it, ++index) {
    if (it->pt() <= etCut_)
      continue;

    l1t::PFCluster cluster(
        it->pt(), it->eta(), it->phi(), /*hOverE=*/0., /*isEM=*/true);  // it->hovere() seems to return random values
    if (corrector_.valid())
      corrector_.correctPt(cluster);
    cluster.setPtError(resol_(cluster.pt(), std::abs(cluster.eta())));
    cluster.setHwQual(it->hwQual());
    out->push_back(cluster);
    out->back().addConstituent(edm::Ptr<l1t::L1Candidate>(clusters, index));
    selector.fill(cluster.pt(), cluster.eta(), cluster.phi(), index);
  }
  std::vector<unsigned int> indices = selector.returnSorted();
  for (unsigned int ii = 0; ii < indices.size(); ii++) {
    unsigned int theIndex = indices[ii];
    l1t::PFCluster cluster((clusters->begin() + theIndex)->pt(),
                           (clusters->begin() + theIndex)->eta(),
                           (clusters->begin() + theIndex)->phi(),
                           /*hOverE=*/0.,
                           /*isEM=*/true);  // it->hovere() seems to return random values
    if (corrector_.valid())
      corrector_.correctPt(cluster);
    cluster.setPtError(resol_(cluster.pt(), std::abs(cluster.eta())));
    cluster.setHwQual((clusters->begin() + theIndex)->hwQual());
    out_sel->push_back(cluster);
    out_sel->back().addConstituent(edm::Ptr<l1t::L1Candidate>(clusters, theIndex));
  }

  iEvent.put(std::move(out), "all");
  iEvent.put(std::move(out_sel), "selected");
}
using l1tpf::PFClusterProducerFromL1EGClusters;
DEFINE_FWK_MODULE(PFClusterProducerFromL1EGClusters);
