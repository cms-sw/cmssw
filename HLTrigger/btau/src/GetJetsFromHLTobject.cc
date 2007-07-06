#include "HLTrigger/btau/interface/GetJetsFromHLTobject.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "DataFormats/Common/interface/RefVector.h"

GetJetsFromHLTobject::GetJetsFromHLTobject(const edm::ParameterSet& iConfig)
{
  //  produces<reco::CaloJetRefVector>();
    produces<reco::CaloJetCollection>();
  m_jetsSrc   = iConfig.getParameter<edm::InputTag>("jets");
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
   iEvent.getByLabel(m_jetsSrc, hltObject);
   for (size_t i = 0; i < hltObject->size(); i++) {
     const CaloJetRef & jetRef = hltObject->getParticleRef(i).castTo<CaloJetRef>();
         //jets->push_back(jetRef);
	 jets->push_back(*jetRef);
   }
   
   iEvent.put(jets);
}
