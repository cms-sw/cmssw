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
   //reco::CaloJetRefVector* jetRefs = new reco::CaloJetRefVector();
      reco::CaloJetCollection* jets = new reco::CaloJetCollection();

   Handle<reco::HLTFilterObjectWithRefs> hltObject;
   iEvent.getByLabel(m_jetsSrc, hltObject);
   for (size_t i = 0; i < hltObject->size(); i++) {
     edm::RefToBase<reco::Candidate> jetCand = hltObject->getParticleRef(i);
     const CaloJetRef& jetRef = jetCand.castTo<CaloJetRef>();
     //jetRefs->push_back(jetRef);
         CaloJet jet(*jetRef);
	 jets->push_back(jet);
   }
   
   //std::auto_ptr<reco::CaloJetRefVector> pJetRefs(jetRefs);
      std::auto_ptr<reco::CaloJetCollection> pJetRefs(jets);
   iEvent.put(pJetRefs);
}
