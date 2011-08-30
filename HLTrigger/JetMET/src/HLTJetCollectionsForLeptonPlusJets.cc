#include "HLTrigger/JetMET/interface/HLTJetCollectionsForLeptonPlusJets.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "TVector3.h"

typedef std::vector<edm::RefVector<std::vector<reco::CaloJet>,reco::CaloJet,edm::refhelper::FindUsingAdvance<std::vector<reco::CaloJet>,reco::CaloJet> > > JetCollectionVector;

HLTJetCollectionsForLeptonPlusJets::HLTJetCollectionsForLeptonPlusJets(const edm::ParameterSet& iConfig):
  hltLeptonTag(iConfig.getParameter< edm::InputTag > ("HltLeptonTag")),
  sourceJetTag(iConfig.getParameter< edm::InputTag > ("SourceJetTag")),
  minDeltaR_(iConfig.getParameter< double > ("minDeltaR"))
{
  produces<JetCollectionVector> ();
}


HLTJetCollectionsForLeptonPlusJets::~HLTJetCollectionsForLeptonPlusJets()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

void HLTJetCollectionsForLeptonPlusJets::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag> ("HltLeptonTag", edm::InputTag("triggerFilterObjectWithRefs"));
    desc.add<edm::InputTag> ("SourceJetTag", edm::InputTag("caloJetCollection"));
    desc.add<double> ("minDeltaR", 0.5);
    descriptions.add("hltJetCollectionsForLeptonPlusJets", desc);
}

//
// member functions
//


// ------------ method called to produce the data  ------------
// template <typename T>
void
HLTJetCollectionsForLeptonPlusJets::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByLabel(hltLeptonTag,PrevFilterOutput);
 
  //its easier on the if statement flow if I try everything at once, shouldnt add to timing
  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > clusCands;
  PrevFilterOutput->getObjects(trigger::TriggerCluster,clusCands);

  std::vector<edm::Ref<reco::ElectronCollection> > eleCands;
  PrevFilterOutput->getObjects(trigger::TriggerElectron,eleCands);
  
  std::vector<reco::RecoChargedCandidateRef> muonCands;
  PrevFilterOutput->getObjects(trigger::TriggerMuon,muonCands);

  //prepare the collection of 3-D vector for lepton momenta
  std::vector<TVector3> LeptonPs;

  if(!clusCands.empty()){ //try trigger cluster
    for(size_t candNr=0;candNr<clusCands.size();candNr++){
      TVector3 positionVector(
                  clusCands[candNr]->superCluster()->position().x(),
                  clusCands[candNr]->superCluster()->position().y(),
                  clusCands[candNr]->superCluster()->position().z());
      LeptonPs.push_back(positionVector);
    }
  }else if(!eleCands.empty()){ // try trigger leptons
    for(size_t candNr=0;candNr<eleCands.size();candNr++){
      TVector3 positionVector(
                  eleCands[candNr]->superCluster()->position().x(),
                  eleCands[candNr]->superCluster()->position().y(),
                  eleCands[candNr]->superCluster()->position().z());
      LeptonPs.push_back(positionVector);
    }
  }else if(!muonCands.empty()){  //try trigger muons
    for(size_t candNr=0;candNr<muonCands.size();candNr++){
      TVector3 positionVector(
                  muonCands[candNr]->px(),
                  muonCands[candNr]->py(),
                  muonCands[candNr]->pz());
      LeptonPs.push_back(positionVector);
    }
  }
  
  edm::Handle<reco::CaloJetCollection> theCaloJetCollectionHandle;
  iEvent.getByLabel(sourceJetTag, theCaloJetCollectionHandle);
  
  const reco::CaloJetCollection & theCaloJetCollection = *theCaloJetCollectionHandle;
  
  std::auto_ptr < JetCollectionVector > allSelections(new JetCollectionVector());
  
    for (unsigned int i = 0; i < LeptonPs.size(); i++) {

        reco::CaloJetRefVector refVector;

        for (unsigned int j = 0; j < theCaloJetCollection.size(); j++) {
            TVector3 JetP(theCaloJetCollection[j].px(), theCaloJetCollection[j].py(), theCaloJetCollection[j].pz());
            double DR = LeptonPs[i].DeltaR(JetP);

            if (DR > minDeltaR_)
        refVector.push_back(reco::CaloJetRef(theCaloJetCollectionHandle, j));
        }
    allSelections->push_back(refVector);
    }

    iEvent.put(allSelections);
  
  return;
  
}

// ------------ method called once each job just before starting event loop  ------------
void HLTJetCollectionsForLeptonPlusJets::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void HLTJetCollectionsForLeptonPlusJets::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTJetCollectionsForLeptonPlusJets);