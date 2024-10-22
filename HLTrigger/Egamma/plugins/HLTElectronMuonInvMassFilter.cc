/** \class HLTElectronMuonInvMassFilter
 *
 *  Original Author: Massimiliano Chiorboli
 *  Institution: INFN, Italy
 *  Contact: Massimiliano.Chiorboli@cern.ch
 *  Date: July 6, 2011
 */

#include "HLTElectronMuonInvMassFilter.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
//
// constructors and destructor
//
HLTElectronMuonInvMassFilter::HLTElectronMuonInvMassFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  eleCandTag_ = iConfig.getParameter<edm::InputTag>("elePrevCandTag");
  muonCandTag_ = iConfig.getParameter<edm::InputTag>("muonPrevCandTag");
  lowerMassCut_ = iConfig.getParameter<double>("lowerMassCut");
  upperMassCut_ = iConfig.getParameter<double>("upperMassCut");
  ncandcut_ = iConfig.getParameter<int>("ncandcut");
  relaxed_ = iConfig.getUntrackedParameter<bool>("electronRelaxed", true);
  L1IsoCollTag_ = iConfig.getParameter<edm::InputTag>("ElectronL1IsoCand");
  L1NonIsoCollTag_ = iConfig.getParameter<edm::InputTag>("ElectronL1NonIsoCand");
  MuonCollTag_ = iConfig.getParameter<edm::InputTag>("MuonCand");
  eleCandToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(eleCandTag_);
  muonCandToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(muonCandTag_);
}

HLTElectronMuonInvMassFilter::~HLTElectronMuonInvMassFilter() = default;

void HLTElectronMuonInvMassFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("elePrevCandTag",
                          edm::InputTag("hltL1NonIsoHLTCaloIdTTrkIdVLSingleElectronEt8NoCandDphiFilter"));
  desc.add<edm::InputTag>("muonPrevCandTag", edm::InputTag("hltL1Mu0HTT50L3Filtered3"));
  desc.add<double>("lowerMassCut", 4.0);
  desc.add<double>("upperMassCut", 999999.0);
  desc.add<int>("ncandcut", 1);
  desc.addUntracked<bool>("electronRelaxed", true);
  desc.add<edm::InputTag>("ElectronL1IsoCand", edm::InputTag("hltPixelMatchElectronsActivity"));
  desc.add<edm::InputTag>("ElectronL1NonIsoCand", edm::InputTag("hltPixelMatchElectronsActivity"));
  desc.add<edm::InputTag>("MuonCand", edm::InputTag("hltL3MuonCandidates"));
  descriptions.add("hltElectronMuonInvMassFilter", desc);
}

// ------------ method called to produce the data  ------------
bool HLTElectronMuonInvMassFilter::hltFilter(edm::Event& iEvent,
                                             const edm::EventSetup& iSetup,
                                             trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace std;
  using namespace math;
  using namespace edm;
  using namespace reco;
  // The filter object
  using namespace trigger;

  if (saveTags()) {
    filterproduct.addCollectionTag(L1IsoCollTag_);
    if (relaxed_)
      filterproduct.addCollectionTag(L1NonIsoCollTag_);
    filterproduct.addCollectionTag(MuonCollTag_);
  }

  edm::Handle<trigger::TriggerFilterObjectWithRefs> EleFromPrevFilter;
  iEvent.getByToken(eleCandToken_, EleFromPrevFilter);

  edm::Handle<trigger::TriggerFilterObjectWithRefs> MuonFromPrevFilter;
  iEvent.getByToken(muonCandToken_, MuonFromPrevFilter);

  vector<Ref<ElectronCollection>> electrons;
  EleFromPrevFilter->getObjects(TriggerElectron, electrons);

  vector<RecoChargedCandidateRef> l3muons;
  MuonFromPrevFilter->getObjects(TriggerMuon, l3muons);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection>> clusCands;
  if (electrons.empty()) {
    EleFromPrevFilter->getObjects(TriggerCluster, clusCands);
  }

  if (clusCands.empty()) {
    EleFromPrevFilter->getObjects(TriggerPhoton, clusCands);
  }

  int nEleMuPairs = 0;

  for (auto& electron : electrons) {
    for (auto& l3muon : l3muons) {
      double mass = (electron->p4() + l3muon->p4()).mass();
      if (mass >= lowerMassCut_ && mass <= upperMassCut_) {
        nEleMuPairs++;
        filterproduct.addObject(TriggerElectron, electron);
        filterproduct.addObject(TriggerMuon, l3muon);
      }
    }
  }

  for (auto& clusCand : clusCands) {
    for (auto& l3muon : l3muons) {
      double mass = (clusCand->p4() + l3muon->p4()).mass();
      if (mass >= lowerMassCut_ && mass <= upperMassCut_) {
        nEleMuPairs++;
        filterproduct.addObject(TriggerElectron, clusCand);
        filterproduct.addObject(TriggerMuon, l3muon);
      }
    }
  }

  // filter decision
  bool accept(nEleMuPairs >= ncandcut_);
  return accept;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTElectronMuonInvMassFilter);
