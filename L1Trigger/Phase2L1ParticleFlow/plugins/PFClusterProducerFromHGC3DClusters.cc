#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Phase2L1ParticleFlow/interface/PFCluster.h"
#include "L1Trigger/Phase2L1ParticleFlow/src/corrector.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/ParametricResolution.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"


namespace l1tpf {
    class PFClusterProducerFromHGC3DClusters : public edm::stream::EDProducer<> {
        public:
            explicit PFClusterProducerFromHGC3DClusters(const edm::ParameterSet&) ;
            ~PFClusterProducerFromHGC3DClusters() {}

        private:
            edm::EDGetTokenT<l1t::HGCalMulticlusterBxCollection> src_;
            bool emOnly_;
            double etCut_;
            bool hasEmId_;
            StringCutObjectSelector<l1t::HGCalMulticluster> emId_;
            l1tpf::corrector corrector_;
            l1tpf::ParametricResolution resol_;

            virtual void produce(edm::Event&, const edm::EventSetup&) override;

    }; // class
} // namespace

l1tpf::PFClusterProducerFromHGC3DClusters::PFClusterProducerFromHGC3DClusters(const edm::ParameterSet & iConfig) :
    src_(consumes<l1t::HGCalMulticlusterBxCollection>(iConfig.getParameter<edm::InputTag>("src"))),
    emOnly_(iConfig.getParameter<bool>("emOnly")),
    etCut_(iConfig.getParameter<double>("etMin")),
    hasEmId_(iConfig.existsAs<std::string>("emId") && !iConfig.getParameter<std::string>("emId").empty()),
    emId_(hasEmId_ ? iConfig.getParameter<std::string>("emId") : ""),
    corrector_(iConfig.getParameter<std::string>("corrector"), 
               emOnly_ || iConfig.getParameter<std::string>("corrector").empty() ? -1 : iConfig.getParameter<double>("correctorEmfMax")),
    resol_(iConfig.getParameter<edm::ParameterSet>("resol"))
{
    produces<l1t::PFClusterCollection>();
    if (hasEmId_) {
        produces<l1t::PFClusterCollection>("em");
        produces<l1t::PFClusterCollection>("had");
    }
}


void 
l1tpf::PFClusterProducerFromHGC3DClusters::produce(edm::Event & iEvent, const edm::EventSetup &) 
{
  std::unique_ptr<l1t::PFClusterCollection> out(new l1t::PFClusterCollection()), outEm, outHad;
  if (hasEmId_) {
      outEm.reset(new l1t::PFClusterCollection());
      outHad.reset(new l1t::PFClusterCollection());
  }
  edm::Handle<l1t::HGCalMulticlusterBxCollection> multiclusters;
  iEvent.getByToken(src_, multiclusters);

  for(auto it = multiclusters->begin(0), ed = multiclusters->end(0); it != ed; ++it) {
      float pt = it->pt(), hoe = it->hOverE();
      bool isEM = hasEmId_ ? emId_(*it) : emOnly_;
      if (emOnly_) { 
          if (hoe == -1) continue;
          pt /= (1 + hoe);
          hoe = 0;
      }
      if (pt <= etCut_) continue;

      l1t::PFCluster cluster(pt, it->eta(), it->phi(), hoe, /*isEM=*/isEM);
      if (corrector_.valid()) corrector_.correctPt(cluster);
      cluster.setPtError(resol_(cluster.pt(), std::abs(cluster.eta())));

      out->push_back(cluster);
      out->back().addConstituent(edm::Ptr<l1t::L1Candidate>(multiclusters, multiclusters->key(it))); 
      if (hasEmId_) {
          (isEM ? outEm : outHad)->push_back(out->back());
      }
  }

  iEvent.put(std::move(out));
  if (hasEmId_) {
      iEvent.put(std::move(outEm), "em");
      iEvent.put(std::move(outHad), "had");
  }
}
using l1tpf::PFClusterProducerFromHGC3DClusters;
DEFINE_FWK_MODULE(PFClusterProducerFromHGC3DClusters);
