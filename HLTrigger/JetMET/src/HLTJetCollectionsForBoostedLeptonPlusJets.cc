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
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "HLTrigger/JetMET/interface/HLTJetCollectionsForBoostedLeptonPlusJets.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include "CommonTools/Utils/interface/PtComparator.h"




template <typename jetType>
HLTJetCollectionsForBoostedLeptonPlusJets<jetType>::HLTJetCollectionsForBoostedLeptonPlusJets(const edm::ParameterSet& iConfig):
  hltLeptonTag(iConfig.getParameter< edm::InputTag > ("HltLeptonTag")),
  sourceJetTag(iConfig.getParameter< edm::InputTag > ("SourceJetTag")),
  minDeltaR_(iConfig.getParameter< double > ("minDeltaR"))
{
  using namespace edm;
  using namespace std;

  typedef vector<RefVector<vector<jetType>,jetType,refhelper::FindUsingAdvance<vector<jetType>,jetType> > > JetCollectionVector;
  typedef vector<jetType> JetCollection;

  m_theLeptonToken = consumes<trigger::TriggerFilterObjectWithRefs>(hltLeptonTag);
  m_theJetToken = consumes<std::vector<jetType>>(sourceJetTag);
  produces<JetCollectionVector>();
  produces<JetCollection>();
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
  typedef edm::Ref<JetCollection> JetRef;
  typedef edm::RefVector<JetCollection> JetRefVector;

  Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken(m_theLeptonToken,PrevFilterOutput);
 
  //its easier on the if statement flow if I try everything at once, shouldnt add to timing
  // Electrons can be stored as objects of types TriggerCluster, TriggerElectron, or TriggerPhoton
  vector<reco::RecoChargedCandidateRef> muonCands;
  PrevFilterOutput->getObjects(trigger::TriggerMuon,muonCands);

  vector<Ref<reco::ElectronCollection> > eleCands;
  PrevFilterOutput->getObjects(trigger::TriggerElectron,eleCands);
  
  trigger::VRphoton photonCands;
  PrevFilterOutput->getObjects(trigger::TriggerPhoton, photonCands);
  
  vector<Ref<reco::RecoEcalCandidateCollection> > clusCands;
  PrevFilterOutput->getObjects(trigger::TriggerCluster,clusCands);

  Handle<JetCollection> theJetCollectionHandle;
  iEvent.getByToken(m_theJetToken, theJetCollectionHandle);
  
  typename JetCollection::const_iterator jet;
  
  auto_ptr<JetCollection> allSelections(new JetCollection);
  auto_ptr<JetCollectionVector> product(new JetCollectionVector);

  std::vector<size_t> usedCands;

  if(!muonCands.empty()){ // muons
    for (jet = theJetCollectionHandle->begin(); jet != theJetCollectionHandle->end(); jet++) {
      //const jetType* referenceJet = &*jet;
      jetType cleanedJet = *jet; //copy original jet
      for(size_t candNr=0;candNr<muonCands.size();candNr++){
        if (std::find(usedCands.begin(),usedCands.end(),candNr)!=usedCands.end()) continue;
        if (deltaR((*muonCands[candNr]),cleanedJet) <= minDeltaR_) {
          std::vector<edm::Ptr<reco::PFCandidate> > pfConstituents = cleanedJet.getPFConstituents();
          for(std::vector<edm::Ptr<reco::PFCandidate> >::const_iterator i_candidate = pfConstituents.begin(); i_candidate != pfConstituents.end(); ++i_candidate){
            if (deltaR((*muonCands[candNr]),(**i_candidate))<0.001) {
	      cleanedJet.setP4( cleanedJet.p4() - muonCands[candNr]->p4());
              usedCands.push_back(candNr);
              break;
            }//if constituent matched
          }//for constituents
        }//if dR<min
      }//for cands
      allSelections->push_back(cleanedJet);
    }//for jets
  }//if cands

  if(!eleCands.empty()){ // electrons
    for (jet = theJetCollectionHandle->begin(); jet != theJetCollectionHandle->end(); jet++) {
      //const jetType* referenceJet = &*jet;
      jetType cleanedJet = *jet; //copy original jet
      for(size_t candNr=0;candNr<eleCands.size();candNr++){
        if (std::find(usedCands.begin(),usedCands.end(),candNr)!=usedCands.end()) continue;
        if (deltaR((*eleCands[candNr]),cleanedJet) <= minDeltaR_) {
          std::vector<edm::Ptr<reco::PFCandidate> > pfConstituents = cleanedJet.getPFConstituents();
          for(std::vector<edm::Ptr<reco::PFCandidate> >::const_iterator i_candidate = pfConstituents.begin(); i_candidate != pfConstituents.end(); ++i_candidate){
            if (deltaR((*eleCands[candNr]),(**i_candidate))<0.001) {
	      cleanedJet.setP4( cleanedJet.p4() - eleCands[candNr]->p4());
              usedCands.push_back(candNr);
              break;
            }//if constituent matched
          }//for constituents
        }//if dR<min
      }//for cands
      allSelections->push_back(cleanedJet);
    }//for jets
  }//if cands

  if(!photonCands.empty()){ // photons
    for (jet = theJetCollectionHandle->begin(); jet != theJetCollectionHandle->end(); jet++) {
      //const jetType* referenceJet = &*jet;
      jetType cleanedJet = *jet; //copy original jet
      for(size_t candNr=0;candNr<photonCands.size();candNr++){
        if (std::find(usedCands.begin(),usedCands.end(),candNr)!=usedCands.end()) continue;
        if (deltaR((*photonCands[candNr]),cleanedJet) <= minDeltaR_) {
          std::vector<edm::Ptr<reco::PFCandidate> > pfConstituents = cleanedJet.getPFConstituents();
          for(std::vector<edm::Ptr<reco::PFCandidate> >::const_iterator i_candidate = pfConstituents.begin(); i_candidate != pfConstituents.end(); ++i_candidate){
            if (deltaR((*photonCands[candNr]),(**i_candidate))<0.001) {
	      cleanedJet.setP4( cleanedJet.p4() - photonCands[candNr]->p4());
              usedCands.push_back(candNr);
              break;
            }//if constituent matched
          }//for constituents
        }//if dR<min
      }//for cands
      allSelections->push_back(cleanedJet);
    }//for jets
  }//if cands

  if(!clusCands.empty()){ // trigger clusters
    for (jet = theJetCollectionHandle->begin(); jet != theJetCollectionHandle->end(); jet++) {
      //const jetType* referenceJet = &*jet;
      jetType cleanedJet = *jet; //copy original jet
      for(size_t candNr=0;candNr<clusCands.size();candNr++){
        if (std::find(usedCands.begin(),usedCands.end(),candNr)!=usedCands.end()) continue;
        if (deltaR((*clusCands[candNr]),cleanedJet) <= minDeltaR_) {
          std::vector<edm::Ptr<reco::PFCandidate> > pfConstituents = cleanedJet.getPFConstituents();
          for(std::vector<edm::Ptr<reco::PFCandidate> >::const_iterator i_candidate = pfConstituents.begin(); i_candidate != pfConstituents.end(); ++i_candidate){
            if (deltaR((*clusCands[candNr]),(**i_candidate))<0.001) {
	      cleanedJet.setP4( cleanedJet.p4() - clusCands[candNr]->p4());
              usedCands.push_back(candNr);
              break;
            }//if constituent matched
          }//for constituents
        }//if dR<min
      }//for cands
      allSelections->push_back(cleanedJet);
    }//for jets
  }//if cands

  NumericSafeGreaterByPt<jetType> compJets;  
  // reorder cleaned jets
  std::sort (allSelections->begin(), allSelections->end(), compJets);
  edm::OrphanHandle<JetCollection> cleanedJetHandle = iEvent.put(allSelections);

  JetCollection const & jets = *cleanedJetHandle;

  JetRefVector cleanedJetRefs;

  for (unsigned iJet = 0; iJet < jets.size(); ++iJet) {
    cleanedJetRefs.push_back(JetRef(cleanedJetHandle, iJet));
  }

  product->emplace_back(cleanedJetRefs);
  iEvent.put(product);

  return;
  
}
