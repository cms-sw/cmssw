#include <cmath>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include "HLTrigger/JetMET/interface/HLTJetCollectionsForLeptonPlusJets.h"

template <typename jetType>
HLTJetCollectionsForLeptonPlusJets<jetType>::HLTJetCollectionsForLeptonPlusJets(const edm::ParameterSet& iConfig)
    : hltLeptonTag(iConfig.getParameter<edm::InputTag>("HltLeptonTag")),
      sourceJetTag(iConfig.getParameter<edm::InputTag>("SourceJetTag")),
      // minimum delta-R^2 threshold with sign
      minDeltaR2_(iConfig.getParameter<double>("minDeltaR") * std::abs(iConfig.getParameter<double>("minDeltaR"))) {
  using namespace edm;
  using namespace std;
  typedef vector<RefVector<vector<jetType>, jetType, refhelper::FindUsingAdvance<vector<jetType>, jetType>>>
      JetCollectionVector;
  m_theLeptonToken = consumes<trigger::TriggerFilterObjectWithRefs>(hltLeptonTag);
  m_theJetToken = consumes<std::vector<jetType>>(sourceJetTag);
  produces<JetCollectionVector>();
}

template <typename jetType>
void HLTJetCollectionsForLeptonPlusJets<jetType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HltLeptonTag", edm::InputTag("triggerFilterObjectWithRefs"));
  desc.add<edm::InputTag>("SourceJetTag", edm::InputTag("caloJetCollection"));
  desc.add<double>("minDeltaR", 0.5);
  descriptions.add(defaultModuleLabel<HLTJetCollectionsForLeptonPlusJets<jetType>>(), desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
// template <typename T>
template <typename jetType>
void HLTJetCollectionsForLeptonPlusJets<jetType>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  typedef vector<RefVector<vector<jetType>, jetType, refhelper::FindUsingAdvance<vector<jetType>, jetType>>>
      JetCollectionVector;
  typedef vector<jetType> JetCollection;
  typedef edm::RefVector<JetCollection> JetRefVector;
  typedef edm::Ref<JetCollection> JetRef;

  Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken(m_theLeptonToken, PrevFilterOutput);

  //its easier on the if statement flow if I try everything at once, shouldnt add to timing
  // Electrons can be stored as objects of types TriggerCluster, TriggerElectron, or TriggerPhoton
  vector<Ref<reco::RecoEcalCandidateCollection>> clusCands;
  PrevFilterOutput->getObjects(trigger::TriggerCluster, clusCands);

  vector<Ref<reco::ElectronCollection>> eleCands;
  PrevFilterOutput->getObjects(trigger::TriggerElectron, eleCands);

  trigger::VRphoton photonCands;
  PrevFilterOutput->getObjects(trigger::TriggerPhoton, photonCands);

  vector<reco::RecoChargedCandidateRef> muonCands;
  PrevFilterOutput->getObjects(trigger::TriggerMuon, muonCands);

  Handle<JetCollection> theJetCollectionHandle;
  iEvent.getByToken(m_theJetToken, theJetCollectionHandle);

  const JetCollection& theJetCollection = *theJetCollectionHandle;

  unique_ptr<JetCollectionVector> allSelections(new JetCollectionVector());

  if (!clusCands.empty()) {  // try trigger clusters
    for (auto const& clusCand : clusCands) {
      JetRefVector refVector;
      for (unsigned int j = 0; j < theJetCollection.size(); j++) {
        if (reco::deltaR2(clusCand->superCluster()->position(), theJetCollection[j]) > minDeltaR2_)
          refVector.push_back(JetRef(theJetCollectionHandle, j));
      }
      allSelections->push_back(refVector);
    }
  }

  if (!eleCands.empty()) {  // try electrons
    for (auto const& eleCand : eleCands) {
      JetRefVector refVector;
      for (unsigned int j = 0; j < theJetCollection.size(); j++) {
        if (reco::deltaR2(eleCand->superCluster()->position(), theJetCollection[j]) > minDeltaR2_)
          refVector.push_back(JetRef(theJetCollectionHandle, j));
      }
      allSelections->push_back(refVector);
    }
  }

  if (!photonCands.empty()) {  // try photons
    for (auto const& photonCand : photonCands) {
      JetRefVector refVector;
      for (unsigned int j = 0; j < theJetCollection.size(); j++) {
        if (reco::deltaR2(photonCand->superCluster()->position(), theJetCollection[j]) > minDeltaR2_)
          refVector.push_back(JetRef(theJetCollectionHandle, j));
      }
      allSelections->push_back(refVector);
    }
  }

  if (!muonCands.empty()) {  // muons
    for (auto const& muonCand : muonCands) {
      JetRefVector refVector;
      for (unsigned int j = 0; j < theJetCollection.size(); j++) {
        if (reco::deltaR2(muonCand->p4(), theJetCollection[j]) > minDeltaR2_)
          refVector.push_back(JetRef(theJetCollectionHandle, j));
      }
      allSelections->push_back(refVector);
    }
  }

  iEvent.put(std::move(allSelections));

  return;
}
