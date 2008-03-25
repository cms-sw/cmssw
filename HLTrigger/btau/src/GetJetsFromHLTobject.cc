#include "HLTrigger/btau/interface/GetJetsFromHLTobject.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/Common/interface/RefVector.h"

GetJetsFromHLTobject::GetJetsFromHLTobject(const edm::ParameterSet& iConfig) :
  m_jets( iConfig.getParameter<edm::InputTag>("jets") )
{
  //produces<reco::CaloJetRefVector>();
  produces<reco::CaloJetCollection>();
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GetJetsFromHLTobject::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;
   std::auto_ptr<reco::CaloJetCollection> jets( new reco::CaloJetCollection() );

   Handle<reco::HLTFilterObjectWithRefs> hltObject;
   iEvent.getByLabel(m_jets, hltObject);
   for (size_t i = 0; i < hltObject->size(); i++) {
     const Candidate * candidate = hltObject->getParticleRef(i).get();
     const CaloJet * jet = dynamic_cast<const reco::CaloJet *>(candidate);
     if (jet)
       jets->push_back(*jet);
     // else 
     //   cerr << ...
   }
   
   iEvent.put(jets);
}
