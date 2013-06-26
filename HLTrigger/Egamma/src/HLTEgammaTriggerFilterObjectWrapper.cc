/** \class HLTEgammaL1MatchFilterRegional
 *
 * $Id: HLTEgammaTriggerFilterObjectWrapper.cc,v 1.2 2012/01/21 14:56:57 fwyzard Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaTriggerFilterObjectWrapper.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

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
}

HLTEgammaTriggerFilterObjectWrapper::~HLTEgammaTriggerFilterObjectWrapper(){}


// ------------ method called to produce the data  ------------
bool HLTEgammaTriggerFilterObjectWrapper::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace trigger;
  using namespace l1extra;

  // Get the recoEcalCandidates
  edm::Handle<reco::RecoEcalCandidateCollection> recoIsolecalcands;
  iEvent.getByLabel(candIsolatedTag_,recoIsolecalcands);
  
  edm::Ref<reco::RecoEcalCandidateCollection> ref;
  // transform the L1Iso_RecoEcalCandidate into the TriggerFilterObjectWithRefs
  for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoIsolecalcands->begin(); recoecalcand!=recoIsolecalcands->end(); recoecalcand++) {
    ref = edm::Ref<reco::RecoEcalCandidateCollection>(recoIsolecalcands, distance(recoIsolecalcands->begin(),recoecalcand) );       
    filterproduct.addObject(TriggerCluster, ref);
  }
  
  if(!doIsolated_) {
    // transform the L1NonIso_RecoEcalCandidate into the TriggerFilterObjectWithRefs and add them to the L1Iso ones.  
    edm::Handle<reco::RecoEcalCandidateCollection> recoNonIsolecalcands;
    iEvent.getByLabel(candNonIsolatedTag_,recoNonIsolecalcands);
    for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoNonIsolecalcands->begin(); recoecalcand!=recoNonIsolecalcands->end(); recoecalcand++) {
      ref = edm::Ref<reco::RecoEcalCandidateCollection>(recoNonIsolecalcands, distance(recoNonIsolecalcands->begin(),recoecalcand) );       
      filterproduct.addObject(TriggerCluster, ref);
    }
  }
  
  return true;  
 }
