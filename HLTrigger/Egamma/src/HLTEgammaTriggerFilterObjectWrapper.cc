/** \class HLTEgammaL1MatchFilterRegional
 *
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaTriggerFilterObjectWrapper.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#define TWOPI 6.283185308
//
// constructors and destructor
//
HLTEgammaTriggerFilterObjectWrapper::HLTEgammaTriggerFilterObjectWrapper(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
   candIsolatedTag_ = iConfig.getParameter< edm::InputTag > ("candIsolatedTag");
   candNonIsolatedTag_ = iConfig.getParameter< edm::InputTag > ("candNonIsolatedTag");
   doIsolated_   = iConfig.getParameter<bool>("doIsolated");
   candIsolatedToken_ = consumes<reco::RecoEcalCandidateCollection>(candIsolatedTag_);
   if(!doIsolated_) candNonIsolatedToken_ = consumes<reco::RecoEcalCandidateCollection>(candNonIsolatedTag_);
}

void
HLTEgammaTriggerFilterObjectWrapper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candIsolatedTag",edm::InputTag("hltL1IsoRecoEcalCandidate"));
  desc.add<edm::InputTag>("candNonIsolatedTag",edm::InputTag("hltL1NonIsoRecoEcalCandidate"));
  desc.add<bool>("doIsolated",false);
  descriptions.add("hltEgammaTriggerFilterObjectWrapper",desc);
}

HLTEgammaTriggerFilterObjectWrapper::~HLTEgammaTriggerFilterObjectWrapper(){}


// ------------ method called to produce the data  ------------
bool HLTEgammaTriggerFilterObjectWrapper::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace trigger;
  using namespace l1extra;

  // Get the recoEcalCandidates
  edm::Handle<reco::RecoEcalCandidateCollection> recoIsolecalcands;
  iEvent.getByToken(candIsolatedToken_,recoIsolecalcands);

  edm::Ref<reco::RecoEcalCandidateCollection> ref;
  // transform the L1Iso_RecoEcalCandidate into the TriggerFilterObjectWithRefs
  for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoIsolecalcands->begin(); recoecalcand!=recoIsolecalcands->end(); recoecalcand++) {
    ref = edm::Ref<reco::RecoEcalCandidateCollection>(recoIsolecalcands, distance(recoIsolecalcands->begin(),recoecalcand) );
    filterproduct.addObject(TriggerCluster, ref);
  }

  if(!doIsolated_) {
    // transform the L1NonIso_RecoEcalCandidate into the TriggerFilterObjectWithRefs and add them to the L1Iso ones.
    edm::Handle<reco::RecoEcalCandidateCollection> recoNonIsolecalcands;
    iEvent.getByToken(candNonIsolatedToken_,recoNonIsolecalcands);
    for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoNonIsolecalcands->begin(); recoecalcand!=recoNonIsolecalcands->end(); recoecalcand++) {
      ref = edm::Ref<reco::RecoEcalCandidateCollection>(recoNonIsolecalcands, distance(recoNonIsolecalcands->begin(),recoecalcand) );
      filterproduct.addObject(TriggerCluster, ref);
    }
  }

  return true;
 }
