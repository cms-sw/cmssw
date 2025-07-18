/** \class HLTEgammaCombMassFilter
 *
 * $Id: HLTEgammaCombMassFilter.cc,
 *
 *  \author Chris Tully (Princeton)
 *
 */

#include "HLTEgammaCombMassFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"

//
// constructors and destructor
//
HLTEgammaCombMassFilter::HLTEgammaCombMassFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  firstLegLastFilterTag_ = iConfig.getParameter<edm::InputTag>("firstLegLastFilter");
  secondLegLastFilterTag_ = iConfig.getParameter<edm::InputTag>("secondLegLastFilter");
  minMass_ = iConfig.getParameter<double>("minMass");
  l1EGTag_ = iConfig.getParameter<edm::InputTag>("l1EGCand");
  firstLegLastFilterToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(firstLegLastFilterTag_);
  secondLegLastFilterToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(secondLegLastFilterTag_);
}

HLTEgammaCombMassFilter::~HLTEgammaCombMassFilter() = default;

void HLTEgammaCombMassFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("firstLegLastFilter", edm::InputTag("firstFilter"));
  desc.add<edm::InputTag>("secondLegLastFilter", edm::InputTag("secondFilter"));
  desc.add<edm::InputTag>("l1EGCand", edm::InputTag("hltEgammaCandidates"));
  desc.add<double>("minMass", -1.0);
  descriptions.add("hltEgammaCombMassFilter", desc);
}

// ------------ method called to produce the data  ------------
bool HLTEgammaCombMassFilter::hltFilter(edm::Event& iEvent,
                                        const edm::EventSetup& iSetup,
                                        trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace trigger;
  if (saveTags()) {
    filterproduct.addCollectionTag(l1EGTag_);
  }

  std::vector<math::XYZTLorentzVector> firstLegP4s;
  std::vector<math::XYZTLorentzVector> secondLegP4s;

  getP4OfLegCands(iEvent, firstLegLastFilterToken_, firstLegP4s);
  getP4OfLegCands(iEvent, secondLegLastFilterToken_, secondLegP4s);

  bool accept = false;
  std::set<math::XYZTLorentzVector, LorentzVectorComparator> addedLegP4s;

  for (size_t i = 0; i < firstLegP4s.size(); i++) {
    for (size_t j = 0; j < secondLegP4s.size(); j++) {
      // Skip if it's the same object
      if (firstLegP4s[i] == secondLegP4s[j])
        continue;

      math::XYZTLorentzVector pairP4 = firstLegP4s[i] + secondLegP4s[j];
      double mass = pairP4.M();
      if (mass >= minMass_) {
        accept = true;

        // Add first leg object if not already added
        if (addedLegP4s.insert(firstLegP4s[i]).second) {
          addObjectToFilterProduct(iEvent, filterproduct, firstLegLastFilterToken_, i);
        }

        // Add second leg object if not already added
        if (addedLegP4s.insert(secondLegP4s[j]).second) {
          addObjectToFilterProduct(iEvent, filterproduct, secondLegLastFilterToken_, j);
        }
      }
    }
  }

  return accept;
}

void HLTEgammaCombMassFilter::addObjectToFilterProduct(
    const edm::Event& iEvent,
    trigger::TriggerFilterObjectWithRefs& filterproduct,
    const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs>& token,
    size_t index) {
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken(token, PrevFilterOutput);

  // Get all types of objects
  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > phoCandsPrev;
  PrevFilterOutput->getObjects(trigger::TriggerPhoton, phoCandsPrev);
  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > clusCandsPrev;
  PrevFilterOutput->getObjects(trigger::TriggerCluster, clusCandsPrev);
  std::vector<edm::Ref<reco::ElectronCollection> > eleCandsPrev;
  PrevFilterOutput->getObjects(trigger::TriggerElectron, eleCandsPrev);

  // Check which type of object corresponds to the given index
  if (index < phoCandsPrev.size()) {
    filterproduct.addObject(trigger::TriggerPhoton, phoCandsPrev[index]);
  } else if (index < phoCandsPrev.size() + clusCandsPrev.size()) {
    filterproduct.addObject(trigger::TriggerCluster, clusCandsPrev[index - phoCandsPrev.size()]);
  } else if (index < phoCandsPrev.size() + clusCandsPrev.size() + eleCandsPrev.size()) {
    filterproduct.addObject(trigger::TriggerElectron, eleCandsPrev[index - phoCandsPrev.size() - clusCandsPrev.size()]);
  } else {
    edm::LogWarning("HLTEgammaCombMassFilter") << "Could not find object at index " << index;
  }
}

void HLTEgammaCombMassFilter::getP4OfLegCands(const edm::Event& iEvent,
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
    for (auto const& phoCand : phoCands) {
      p4s.push_back(phoCand->p4());
    }
  } else if (!clusCands.empty()) {
    //try trigger cluster (should never be this, at the time of writing (17/1/11) this would indicate an error)
    for (auto const& clusCand : clusCands) {
      p4s.push_back(clusCand->p4());
    }
  } else if (!eleCands.empty()) {
    for (auto const& eleCand : eleCands) {
      p4s.push_back(eleCand->p4());
    }
  }
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTEgammaCombMassFilter);
