#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/Common/interface/ValueMap.h"
class PUFilter : public edm::global::EDProducer <> {
   public:
      explicit PUFilter(const edm::ParameterSet&);

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      const edm::EDGetTokenT<edm::View<reco::PFJet> > jetsToken_;
      const edm::EDGetTokenT<edm::ValueMap<int> > jetPuIdToken_;
      virtual void produce(edm::StreamID , edm::Event& , const edm::EventSetup & ) const override;
     };


PUFilter::PUFilter(const edm::ParameterSet& iConfig):
jetsToken_( consumes<edm::View<reco::PFJet> > (iConfig.getParameter<edm::InputTag>("Jets") ) ),
jetPuIdToken_( consumes<edm::ValueMap<int> > (iConfig.getParameter<edm::InputTag>("JetPUID") ) )
{
   produces<std::vector<reco::PFJet> > ();
}


void
PUFilter::produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup & iSetup) const
{
   using namespace edm;

  Handle<edm::View<reco::PFJet> > jetsH;
  Handle< edm::ValueMap<int> > id_decisions;
  
  iEvent.getByToken( jetsToken_, jetsH );
  iEvent.getByToken( jetPuIdToken_, id_decisions );
  
  std::auto_ptr<std::vector<reco::PFJet> > goodjets(new std::vector<reco::PFJet> );
  for( size_t i = 0; i < jetsH->size(); ++i ) {
    auto jet = jetsH->refAt(i);
    if((*id_decisions)[jet]) goodjets->push_back(*jet);
  }
  iEvent.put(goodjets);
}

void
PUFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("Jets", edm::InputTag("hltAK4PFJetsCorrected"));
  desc.add<edm::InputTag>("JetPUID", edm::InputTag("MVAJetPuIdProducer","CATEv0Id"));
  descriptions.add("PUFilter",desc);
  desc.setUnknown();
}
DEFINE_FWK_MODULE(PUFilter);
