#include "HLTrigger/JetMET/interface/HLTJetCollectionsForElePlusJets.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "TVector3.h"

typedef std::vector<edm::RefVector<std::vector<reco::CaloJet>,reco::CaloJet,edm::refhelper::FindUsingAdvance<std::vector<reco::CaloJet>,reco::CaloJet> > > JetCollectionVector;

HLTJetCollectionsForElePlusJets::HLTJetCollectionsForElePlusJets(const edm::ParameterSet& iConfig):
  hltElectronTag(iConfig.getParameter< edm::InputTag > ("HltElectronTag")),
  sourceJetTag(iConfig.getParameter< edm::InputTag > ("SourceJetTag")),
  //minJetPt_(iConfig.getParameter<double> ("MinJetPt")),
  //maxAbsJetEta_(iConfig.getParameter<double> ("MaxAbsJetEta")),
  //minNJets_(iConfig.getParameter<unsigned int> ("MinNJets")),
  minDeltaR_(iConfig.getParameter< double > ("minDeltaR"))
  //Only for VBF
  //minSoftJetPt_(iConfig.getParameter< double > ("MinSoftJetPt")),
  //minDeltaEta_(iConfig.getParameter< double > ("MinDeltaEta"))
{
  produces<JetCollectionVector> ();
}


HLTJetCollectionsForElePlusJets::~HLTJetCollectionsForElePlusJets()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

void HLTJetCollectionsForElePlusJets::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag> ("HltElectronTag", edm::InputTag("triggerFilterObjectWithRefs"));
    desc.add<edm::InputTag> ("SourceJetTag", edm::InputTag("caloJetCollection"));
   // desc.add<double> ("MinJetPt", 30.);
   // desc.add<double> ("MaxAbsJetEta", 2.6);
   // desc.add<unsigned int> ("MinNJets", 1);
    desc.add<double> ("minDeltaR", 0.5);
    //Only for VBF
   // desc.add<double> ("MinSoftJetPt", 25.);
    //desc.add<double> ("MinDeltaEta", -1.);
    descriptions.add("hltJetCollectionsForElePlusJets", desc);
}

//
// member functions
//


// ------------ method called to produce the data  ------------
// template <typename T>
void
HLTJetCollectionsForElePlusJets::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByLabel(hltElectronTag,PrevFilterOutput);
 
  //its easier on the if statement flow if I try everything at once, shouldnt add to timing
  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > clusCands;
  PrevFilterOutput->getObjects(trigger::TriggerCluster,clusCands);

  std::vector<edm::Ref<reco::ElectronCollection> > eleCands;
  PrevFilterOutput->getObjects(trigger::TriggerElectron,eleCands);
  
  //prepare the collection of 3-D vector for electron momenta
  std::vector<TVector3> ElePs;

  if(!clusCands.empty()){ //try trigger cluster
    for(size_t candNr=0;candNr<clusCands.size();candNr++){
      TVector3 positionVector(
                  clusCands[candNr]->superCluster()->position().x(),
                  clusCands[candNr]->superCluster()->position().y(),
                  clusCands[candNr]->superCluster()->position().z());
      ElePs.push_back(positionVector);
    }
  }else if(!eleCands.empty()){ // try trigger electrons
    for(size_t candNr=0;candNr<eleCands.size();candNr++){
      TVector3 positionVector(
                  eleCands[candNr]->superCluster()->position().x(),
                  eleCands[candNr]->superCluster()->position().y(),
                  eleCands[candNr]->superCluster()->position().z());
      ElePs.push_back(positionVector);
    }
  }
  
  edm::Handle<reco::CaloJetCollection> theCaloJetCollectionHandle;
  iEvent.getByLabel(sourceJetTag, theCaloJetCollectionHandle);
  //const reco::CaloJetCollection* theCaloJetCollection = theCaloJetCollectionHandle.product();
  
  const reco::CaloJetCollection & theCaloJetCollection = *theCaloJetCollectionHandle;
  
  //std::auto_ptr< reco::CaloJetCollection >  theFilteredCaloJetCollection(new reco::CaloJetCollection);
  
  std::auto_ptr < JetCollectionVector > allSelections(new JetCollectionVector());
  
 //bool foundSolution(false);

    for (unsigned int i = 0; i < ElePs.size(); i++) {

       // bool VBFJetPair = false;
        //std::vector<int> store_jet;
        reco::CaloJetRefVector refVector;

        for (unsigned int j = 0; j < theCaloJetCollection.size(); j++) {
            TVector3 JetP(theCaloJetCollection[j].px(), theCaloJetCollection[j].py(), theCaloJetCollection[j].pz());
            double DR = ElePs[i].DeltaR(JetP);

            if (DR > minDeltaR_)
        refVector.push_back(reco::CaloJetRef(theCaloJetCollectionHandle, j));
        }
    allSelections->push_back(refVector);
    }

    //iEvent.put(theFilteredCaloJetCollection);
    iEvent.put(allSelections);
  
  return;
  
}

// ------------ method called once each job just before starting event loop  ------------
void HLTJetCollectionsForElePlusJets::beginJob() {
}

// ------------ method called once each job just after ending the event loop  ------------
void HLTJetCollectionsForElePlusJets::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTJetCollectionsForElePlusJets);
