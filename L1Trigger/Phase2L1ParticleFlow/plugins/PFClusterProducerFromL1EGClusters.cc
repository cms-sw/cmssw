#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"
#include "L1Trigger/Phase2L1ParticleFlow/src/corrector.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/ParametricResolution.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"

namespace l1tpf {
  class PFClusterProducerFromL1EGClusters : public edm::stream::EDProducer<> {
  public:
    explicit PFClusterProducerFromL1EGClusters(const edm::ParameterSet &);
    ~PFClusterProducerFromL1EGClusters() override {}

  private:
    edm::EDGetTokenT<BXVector<l1t::EGamma>> src_;
    double etCut_;
    l1tpf::corrector corrector_;
    l1tpf::ParametricResolution resol_;

    void produce(edm::Event &, const edm::EventSetup &) override;

  };  // class
}  // namespace l1tpf

l1tpf::PFClusterProducerFromL1EGClusters::PFClusterProducerFromL1EGClusters(const edm::ParameterSet &iConfig)
    : src_(consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<edm::InputTag>("src"))),
      etCut_(iConfig.getParameter<double>("etMin")),
      corrector_(iConfig.getParameter<std::string>("corrector"), -1),
      resol_(iConfig.getParameter<edm::ParameterSet>("resol")) {
  produces<l1t::PFClusterCollection>();
}

void l1tpf::PFClusterProducerFromL1EGClusters::produce(edm::Event &iEvent, const edm::EventSetup &) {
  std::unique_ptr<l1t::PFClusterCollection> out(new l1t::PFClusterCollection());
  edm::Handle<BXVector<l1t::EGamma>> clusters;
  iEvent.getByToken(src_, clusters);

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
  }

  iEvent.put(std::move(out));
}
using l1tpf::PFClusterProducerFromL1EGClusters;
DEFINE_FWK_MODULE(PFClusterProducerFromL1EGClusters);
