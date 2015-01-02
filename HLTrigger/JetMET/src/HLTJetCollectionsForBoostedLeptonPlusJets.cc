#include <string>
#include <vector>

#include "TVector3.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "HLTrigger/JetMET/interface/HLTJetCollectionsForBoostedLeptonPlusJets.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"




template <typename jetType>
HLTJetCollectionsForBoostedLeptonPlusJets<jetType>::HLTJetCollectionsForBoostedLeptonPlusJets(const edm::ParameterSet& iConfig):
  hltLeptonTag(iConfig.getParameter< edm::InputTag > ("HltLeptonTag")),
  sourceJetTag(iConfig.getParameter< edm::InputTag > ("SourceJetTag")),
  minDeltaR_(iConfig.getParameter< double > ("minDeltaR"))
{
  using namespace edm;
  using namespace std;
  typedef vector<RefVector<vector<jetType>,jetType,refhelper::FindUsingAdvance<vector<jetType>,jetType> > > JetCollectionVector;
  m_theLeptonToken = consumes<trigger::TriggerFilterObjectWithRefs>(hltLeptonTag);
  m_theJetToken = consumes<std::vector<jetType>>(sourceJetTag);
  produces<JetCollectionVector> ();
}

template <typename jetType>
HLTJetCollectionsForBoostedLeptonPlusJets<jetType>::~HLTJetCollectionsForBoostedLeptonPlusJets()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

template <typename jetType>
void
HLTJetCollectionsForBoostedLeptonPlusJets<jetType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag> ("HltLeptonTag", edm::InputTag("triggerFilterObjectWithRefs"));
    //(2)
    desc.add<edm::InputTag> ("SourceJetTag", edm::InputTag("jetCollection"));
    //(2)
    desc.add<double> ("minDeltaR", 0.5);
    descriptions.add(defaultModuleLabel<HLTJetCollectionsForBoostedLeptonPlusJets<jetType>>(), desc);
}

//
// member functions
//


// ------------ method called to produce the data  ------------
// template <typename T>
template <typename jetType>
void
HLTJetCollectionsForBoostedLeptonPlusJets<jetType>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;
  //(3)
  using namespace reco;
  //(3)
  
  typedef vector<RefVector<vector<jetType>,jetType,refhelper::FindUsingAdvance<vector<jetType>,jetType> > > JetCollectionVector;
  typedef vector<jetType> JetCollection;
  typedef edm::RefVector<JetCollection> JetRefVector;
  typedef edm::Ref<JetCollection> JetRef;
  //(4)
  typedef math::XYZTLorentzVector LorentzVector;
  //(4)

  Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken(m_theLeptonToken,PrevFilterOutput);
 
  //its easier on the if statement flow if I try everything at once, shouldnt add to timing
  // Electrons can be stored as objects of types TriggerCluster, TriggerElectron, or TriggerPhoton
  vector<Ref<reco::RecoEcalCandidateCollection> > clusCands;
  PrevFilterOutput->getObjects(trigger::TriggerCluster,clusCands);

  vector<Ref<reco::ElectronCollection> > eleCands;
  PrevFilterOutput->getObjects(trigger::TriggerElectron,eleCands);
  
  trigger::VRphoton photonCands;
  PrevFilterOutput->getObjects(trigger::TriggerPhoton, photonCands);
  
  vector<reco::RecoChargedCandidateRef> muonCands;
  PrevFilterOutput->getObjects(trigger::TriggerMuon,muonCands);

  Handle<JetCollection> theJetCollectionHandle;
  iEvent.getByToken(m_theJetToken, theJetCollectionHandle);
  
  const JetCollection & theJetCollection = *theJetCollectionHandle;
  
  auto_ptr < JetCollectionVector > allSelections(new JetCollectionVector());
  
 if(!clusCands.empty()){ // try trigger clusters
    for(size_t candNr=0;candNr<clusCands.size();candNr++){  
        JetRefVector refVector;
        for (unsigned int j = 0; j < theJetCollection.size(); j++) {
          if (deltaR(clusCands[candNr]->superCluster()->position(),theJetCollection[j]) > minDeltaR_) refVector.push_back(JetRef(theJetCollectionHandle, j));
         else{
        unsigned int w =0 ; 
        std::vector<reco::PFCandidatePtr> pfConstituents = theJetCollection[j].getPFConstituents();
        for(std::vector<reco::PFCandidatePtr>::const_iterator i_candidate = pfConstituents.begin(); i_candidate != pfConstituents.end(); ++i_candidate){
		   TVector3 ClusP(clusCands[candNr]->p4().Px(),clusCands[candNr]->p4().Py(), clusCands[candNr]->p4().Pz());
		   TVector3 PFJetConstP((*i_candidate)->px(),(*i_candidate)->py(),(*i_candidate)->pz());
           double deltaRPFConste = ClusP.DeltaR(PFJetConstP);  
           if(deltaRPFConste < 0.001 && w==0){
		   const_cast<LorentzVector&>(theJetCollection[j].p4()) = theJetCollection[j].p4() - clusCands[candNr]->p4();
            w ++;
           } //if
          }//for constituents
        refVector.push_back(JetRef(theJetCollectionHandle, j));
       }//else
   }
    allSelections->push_back(refVector);
    }
 }

 if(!eleCands.empty()){ // try electrons
    for(size_t candNr=0;candNr<eleCands.size();candNr++){  
        JetRefVector refVector;
        for (unsigned int j = 0; j < theJetCollection.size(); j++) {
          if (deltaR(eleCands[candNr]->superCluster()->position(),theJetCollection[j]) > minDeltaR_) refVector.push_back(JetRef(theJetCollectionHandle, j));
           else{
        unsigned int w =0 ; 
        std::vector<reco::PFCandidatePtr> pfConstituents = theJetCollection[j].getPFConstituents();
        for(std::vector<reco::PFCandidatePtr>::const_iterator i_candidate = pfConstituents.begin(); i_candidate != pfConstituents.end(); ++i_candidate){
		   TVector3 ElectronP(eleCands[candNr]->p4().Px(),eleCands[candNr]->p4().Py(), eleCands[candNr]->p4().Pz());
		   TVector3 PFJetConstP((*i_candidate)->px(),(*i_candidate)->py(),(*i_candidate)->pz());
           double deltaRPFConste = ElectronP.DeltaR(PFJetConstP);  
           if(deltaRPFConste < 0.001 && w==0){
		   const_cast<LorentzVector&>(theJetCollection[j].p4()) = theJetCollection[j].p4() - eleCands[candNr]->p4();
            w ++;
           } //if
          }//for constituents
        refVector.push_back(JetRef(theJetCollectionHandle, j));
       }//else
        }//for jet collection
          
    allSelections->push_back(refVector);
    }
 }
 
 if(!photonCands.empty()){ // try photons
    for(size_t candNr=0;candNr<photonCands.size();candNr++){  
        JetRefVector refVector;
        for (unsigned int j = 0; j < theJetCollection.size(); j++) {
          if (deltaR(photonCands[candNr]->superCluster()->position(),theJetCollection[j]) > minDeltaR_) refVector.push_back(JetRef(theJetCollectionHandle, j));
          else{
        unsigned int w =0 ; 
        std::vector<reco::PFCandidatePtr> pfConstituents = theJetCollection[j].getPFConstituents();
        for(std::vector<reco::PFCandidatePtr>::const_iterator i_candidate = pfConstituents.begin(); i_candidate != pfConstituents.end(); ++i_candidate){
		   TVector3 PhotonP(photonCands[candNr]->p4().Px(),photonCands[candNr]->p4().Py(), photonCands[candNr]->p4().Pz());
		   TVector3 PFJetConstP((*i_candidate)->px(),(*i_candidate)->py(),(*i_candidate)->pz());
           double deltaRPFConste = PhotonP.DeltaR(PFJetConstP);  
           if(deltaRPFConste < 0.001 && w==0){
		   const_cast<LorentzVector&>(theJetCollection[j].p4()) = theJetCollection[j].p4() - photonCands[candNr]->p4();
            w ++;
           } //if
          }//for constituents
        refVector.push_back(JetRef(theJetCollectionHandle, j));
       }//else
        }//for jet collection
    allSelections->push_back(refVector);
    }
 }

 if(!muonCands.empty()){ // muons
    for(size_t candNr=0;candNr<muonCands.size();candNr++){  
        JetRefVector refVector;
        for (unsigned int j = 0; j < theJetCollection.size(); j++) {
	  if (deltaR(muonCands[candNr]->p4(),theJetCollection[j]) > minDeltaR_) refVector.push_back(JetRef(theJetCollectionHandle, j));
      else{
        unsigned int w =0 ; 
        std::vector<reco::PFCandidatePtr> pfConstituents = theJetCollection[j].getPFConstituents();
        for(std::vector<reco::PFCandidatePtr>::const_iterator i_candidate = pfConstituents.begin(); i_candidate != pfConstituents.end(); ++i_candidate){
		   TVector3 MuP(muonCands[candNr]->p4().Px(),muonCands[candNr]->p4().Py(), muonCands[candNr]->p4().Pz());
		   TVector3 PFJetConstP((*i_candidate)->px(),(*i_candidate)->py(),(*i_candidate)->pz());
           double deltaRPFConste = MuP.DeltaR(PFJetConstP);  
           if(deltaRPFConste < 0.001 && w==0){
		   const_cast<LorentzVector&>(theJetCollection[j].p4()) = theJetCollection[j].p4() - muonCands[candNr]->p4();
            w ++;
           } //if
          }//for constituents
        refVector.push_back(JetRef(theJetCollectionHandle, j));
       }//else
        }
    allSelections->push_back(refVector);
    }
 }




 iEvent.put(allSelections);
  
  return;
  
}

