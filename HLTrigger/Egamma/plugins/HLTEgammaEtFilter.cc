/** \class HLTEgammaEtFilter
 *
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTEgammaEtFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

//
// constructors and destructor
//
HLTEgammaEtFilter::HLTEgammaEtFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  inputTag_ = iConfig.getParameter<edm::InputTag>("inputTag");
  etcutEB_ = iConfig.getParameter<double>("etcutEB");
  etcutEE_ = iConfig.getParameter<double>("etcutEE");
  minEtaCut_ = iConfig.getParameter<double>("minEtaCut");
  maxEtaCut_ = iConfig.getParameter<double>("maxEtaCut");
  ncandcut_ = iConfig.getParameter<int>("ncandcut");
  l1EGTag_ = iConfig.getParameter<edm::InputTag>("l1EGCand");
  inputToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(inputTag_);
}

void HLTEgammaEtFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag", edm::InputTag("HLTEgammaL1MatchFilter"));
  desc.add<edm::InputTag>("l1EGCand", edm::InputTag("hltL1IsoRecoEcalCandidate"));
  desc.add<double>("etcutEB", 1.0);
  desc.add<double>("etcutEE", 1.0);
  desc.add<double>("minEtaCut", -9999.0);
  desc.add<double>("maxEtaCut", 9999.0);
  desc.add<int>("ncandcut", 1);
  descriptions.add("hltEgammaEtFilter", desc);
}

HLTEgammaEtFilter::~HLTEgammaEtFilter() = default;

// ------------ method called to produce the data  ------------
bool HLTEgammaEtFilter::hltFilter(edm::Event& iEvent,
                                  const edm::EventSetup& iSetup,
                                  trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace trigger;

  // The filter object
  if (saveTags()) {
    filterproduct.addCollectionTag(l1EGTag_);
    ;
  }

  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;

  // get hold of filtered candidates
  //edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken(inputToken_, PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;  // vref with your specific C++ collection type
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
  if (recoecalcands.empty())
    PrevFilterOutput->getObjects(TriggerPhoton,
                                 recoecalcands);  //we dont know if its type trigger cluster or trigger photon

  // look at all candidates,  check cuts and add to filter object
  int n(0);

  for (auto& recoecalcand : recoecalcands) {
    ref = recoecalcand;

    if ((ref->eta() < minEtaCut_) or (ref->eta() > maxEtaCut_))
      continue;

    if ((fabs(ref->eta()) < 1.479 && ref->et() >= etcutEB_) || (fabs(ref->eta()) >= 1.479 && ref->et() >= etcutEE_)) {
      n++;
      // std::cout << "Passed eta: " << ref->eta() << std::endl;
      filterproduct.addObject(TriggerCluster, ref);
    }
  }

  // filter decision
  bool accept(n >= ncandcut_);

  return accept;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTEgammaEtFilter);
