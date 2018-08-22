#include <string>
#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/RefToPtr.h" 
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace pat {
  class UnclusteredBlobProducer : public edm::global::EDProducer<>{
  public:
    explicit UnclusteredBlobProducer(const edm::ParameterSet&);
    ~UnclusteredBlobProducer() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  private:
    edm::EDGetTokenT<edm::View<reco::Candidate> > candsrc_;
  };
}


pat::UnclusteredBlobProducer::UnclusteredBlobProducer(const edm::ParameterSet& iConfig) :
  candsrc_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("candsrc") ))
{
  produces<std::vector<reco::VertexCompositePtrCandidate>>();
}

pat::UnclusteredBlobProducer::~UnclusteredBlobProducer() {}

void pat::UnclusteredBlobProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  
  auto blob = std::make_unique<std::vector<reco::VertexCompositePtrCandidate>>();
  
  edm::Handle<edm::View<reco::Candidate>> candidates;
  iEvent.getByToken(candsrc_, candidates);
  
  blob->emplace_back();

  auto& c_blob = blob->back();

  // combine all candidates into composite so they can be accessed properly by CandPtrSelector
  for(unsigned i = 0; i < candidates->size(); ++i){
    c_blob.addDaughter(candidates->ptrAt(i));
  }

  iEvent.put(std::move(blob));
  
}

void pat::UnclusteredBlobProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("candsrc",edm::InputTag("badUnclustered"));

  descriptions.add("UnclusteredBlobProducer",desc);
}

using pat::UnclusteredBlobProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(UnclusteredBlobProducer);
