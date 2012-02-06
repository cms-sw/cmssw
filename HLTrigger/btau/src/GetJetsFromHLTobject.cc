
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "GetJetsFromHLTobject.h"

GetJetsFromHLTobject::GetJetsFromHLTobject(const edm::ParameterSet& iConfig) :
  m_jets( iConfig.getParameter<edm::InputTag>("jets") )
{
  produces<reco::CaloJetCollection>();
}


void
GetJetsFromHLTobject::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;
   std::auto_ptr<reco::CaloJetCollection> jets( new reco::CaloJetCollection() );

   Handle<trigger::TriggerFilterObjectWithRefs> hltObject;
   iEvent.getByLabel(m_jets, hltObject);
   std::vector<reco::CaloJetRef> refs;
   hltObject->getObjects( trigger::TriggerBJet, refs );
   for (size_t i = 0; i < refs.size(); i++) {
     jets->push_back(* refs[i]);
   }
   
   iEvent.put(jets);
}
