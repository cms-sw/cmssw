#include "HLTrigger/JetMET/interface/HLTJetCollectionsForLeptonPlusJets.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Math/interface/deltaR.h"




template <typename jetType>
HLTJetCollectionsForLeptonPlusJets<jetType>::HLTJetCollectionsForLeptonPlusJets(const edm::ParameterSet& iConfig):
  hltLeptonTag(iConfig.getParameter< edm::InputTag > ("HltLeptonTag")),
  sourceJetTag(iConfig.getParameter< edm::InputTag > ("SourceJetTag")),
  minDeltaR_(iConfig.getParameter< double > ("minDeltaR"))
{
  using namespace edm;
  using namespace std;
  typedef vector<RefVector<vector<jetType>,jetType,refhelper::FindUsingAdvance<vector<jetType>,jetType> > > JetCollectionVector;
  produces<JetCollectionVector> ();
}

template <typename jetType>
HLTJetCollectionsForLeptonPlusJets<jetType>::~HLTJetCollectionsForLeptonPlusJets()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

template <typename jetType>
void
HLTJetCollectionsForLeptonPlusJets<jetType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag> ("HltLeptonTag", edm::InputTag("triggerFilterObjectWithRefs"));
    desc.add<edm::InputTag> ("SourceJetTag", edm::InputTag("caloJetCollection"));
    desc.add<double> ("minDeltaR", 0.5);
    descriptions.add(std::string("hlt")+std::string(typeid(HLTJetCollectionsForLeptonPlusJets<jetType>).name()),desc);
}

//
// member functions
//


// ------------ method called to produce the data  ------------
// template <typename T>
template <typename jetType>
void
HLTJetCollectionsForLeptonPlusJets<jetType>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;
  
  typedef vector<RefVector<vector<jetType>,jetType,refhelper::FindUsingAdvance<vector<jetType>,jetType> > > JetCollectionVector;
  typedef vector<jetType> JetCollection;
  typedef edm::RefVector<JetCollection> JetRefVector;
  typedef edm::Ref<JetCollection> JetRef;

  Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByLabel(hltLeptonTag,PrevFilterOutput);
 
  //its easier on the if statement flow if I try everything at once, shouldnt add to timing
  vector<Ref<reco::RecoEcalCandidateCollection> > clusCands;
  PrevFilterOutput->getObjects(trigger::TriggerCluster,clusCands);

  vector<Ref<reco::ElectronCollection> > eleCands;
  PrevFilterOutput->getObjects(trigger::TriggerElectron,eleCands);
  
  vector<reco::RecoChargedCandidateRef> muonCands;
  PrevFilterOutput->getObjects(trigger::TriggerMuon,muonCands);

  Handle<JetCollection> theJetCollectionHandle;
  iEvent.getByLabel(sourceJetTag, theJetCollectionHandle);
  
  const JetCollection & theJetCollection = *theJetCollectionHandle;
  
  auto_ptr < JetCollectionVector > allSelections(new JetCollectionVector());
  
 if(!clusCands.empty()){ //try trigger cluster
    for(size_t candNr=0;candNr<clusCands.size();candNr++){  
        JetRefVector refVector;
        for (unsigned int j = 0; j < theJetCollection.size(); j++) {
          if (deltaR(clusCands[candNr]->superCluster()->position(),theJetCollection[j]) > minDeltaR_) refVector.push_back(JetRef(theJetCollectionHandle, j));
        }
    allSelections->push_back(refVector);
    }
 }

 if(!eleCands.empty()){ //try trigger cluster
    for(size_t candNr=0;candNr<eleCands.size();candNr++){  
        JetRefVector refVector;
        for (unsigned int j = 0; j < theJetCollection.size(); j++) {
          if (deltaR(eleCands[candNr]->superCluster()->position(),theJetCollection[j]) > minDeltaR_) refVector.push_back(JetRef(theJetCollectionHandle, j));
        }
    allSelections->push_back(refVector);
    }
 }

 if(!muonCands.empty()){ //try trigger cluster
    for(size_t candNr=0;candNr<muonCands.size();candNr++){  
        JetRefVector refVector;
        for (unsigned int j = 0; j < theJetCollection.size(); j++) {
	  if (deltaR(muonCands[candNr]->p4(),theJetCollection[j]) > minDeltaR_) refVector.push_back(JetRef(theJetCollectionHandle, j));
        }
    allSelections->push_back(refVector);
    }
 }




 iEvent.put(allSelections);
  
  return;
  
}

