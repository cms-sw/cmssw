/** \class HLTEgammaAllCombMassFilter
 *
 * $Id: HLTEgammaAllCombMassFilter.cc,
 *
 *  \author Chris Tully (Princeton)
 *
 */

#include "HLTEgammaAllCombMassFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"

//
// constructors and destructor
//
HLTEgammaAllCombMassFilter::HLTEgammaAllCombMassFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  firstLegLastFilterTag_ = iConfig.getParameter<edm::InputTag>("firstLegLastFilter");
  secondLegLastFilterTag_ = iConfig.getParameter<edm::InputTag>("secondLegLastFilter");
  minMass_ = iConfig.getParameter<double>("minMass");
  firstLegLastFilterToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(firstLegLastFilterTag_);
  secondLegLastFilterToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(secondLegLastFilterTag_);
}

HLTEgammaAllCombMassFilter::~HLTEgammaAllCombMassFilter() = default;

void HLTEgammaAllCombMassFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("firstLegLastFilter", edm::InputTag("firstFilter"));
  desc.add<edm::InputTag>("secondLegLastFilter", edm::InputTag("secondFilter"));
  desc.add<double>("minMass", -1.0);
  descriptions.add("hltEgammaAllCombMassFilter", desc);
}

// ------------ method called to produce the data  ------------

bool HLTEgammaAllCombMassFilter::hltFilter(edm::Event& iEvent,
                                           const edm::EventSetup& iSetup,
                                           trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  //right, issue 1, we dont know if this is a TriggerElectron, TriggerPhoton, TriggerCluster (should never be a TriggerCluster btw as that implies the 4-vectors are not stored in AOD)

  //trigger::TriggerObjectType firstLegTrigType;
  std::vector<math::XYZTLorentzVector> firstLegP4s;

  //trigger::TriggerObjectType secondLegTrigType;
  std::vector<math::XYZTLorentzVector> secondLegP4s;

  math::XYZTLorentzVector pairP4;

  getP4OfLegCands(iEvent, firstLegLastFilterToken_, firstLegP4s);
  getP4OfLegCands(iEvent, secondLegLastFilterToken_, secondLegP4s);

  bool accept = false;
  for (auto& firstLegP4 : firstLegP4s) {
    for (auto& secondLegP4 : secondLegP4s) {
      math::XYZTLorentzVector pairP4 = firstLegP4 + secondLegP4;
      double mass = pairP4.M();
      if (mass >= minMass_)
        accept = true;
    }
  }
  for (size_t firstLegNr = 0; firstLegNr < firstLegP4s.size(); firstLegNr++) {
    for (size_t secondLegNr = 0; secondLegNr < firstLegP4s.size(); secondLegNr++) {
      math::XYZTLorentzVector pairP4 = firstLegP4s[firstLegNr] + firstLegP4s[secondLegNr];
      double mass = pairP4.M();
      if (mass >= minMass_)
        accept = true;
    }
  }
  for (size_t firstLegNr = 0; firstLegNr < secondLegP4s.size(); firstLegNr++) {
    for (size_t secondLegNr = 0; secondLegNr < secondLegP4s.size(); secondLegNr++) {
      math::XYZTLorentzVector pairP4 = secondLegP4s[firstLegNr] + secondLegP4s[secondLegNr];
      double mass = pairP4.M();
      if (mass >= minMass_)
        accept = true;
    }
  }

  return accept;
}

void HLTEgammaAllCombMassFilter::getP4OfLegCands(
    const edm::Event& iEvent,
    const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>& filterToken,
    std::vector<math::XYZTLorentzVector>& p4s) {
  edm::Handle<trigger::TriggerFilterObjectWithRefs> filterOutput;
  iEvent.getByToken(filterToken, filterOutput);

  //its easier on the if statement flow if I try everything at once, shouldnt add to timing
  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > phoCands;
  filterOutput->getObjects(trigger::TriggerPhoton, phoCands);
  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > clusCands;
  filterOutput->getObjects(trigger::TriggerCluster, clusCands);
  std::vector<edm::Ref<reco::ElectronCollection> > eleCands;
  filterOutput->getObjects(trigger::TriggerElectron, eleCands);

  if (!phoCands.empty()) {  //its photons
    for (auto& phoCand : phoCands) {
      p4s.push_back(phoCand->p4());
    }
  } else if (!clusCands.empty()) {
    //try trigger cluster (should never be this, at the time of writing (17/1/11) this would indicate an error)
    for (auto& clusCand : clusCands) {
      p4s.push_back(clusCand->p4());
    }
  } else if (!eleCands.empty()) {
    for (auto& eleCand : eleCands) {
      p4s.push_back(eleCand->p4());
    }
  }
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTEgammaAllCombMassFilter);
