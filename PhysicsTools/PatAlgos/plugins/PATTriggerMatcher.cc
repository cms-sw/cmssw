//
// $Id: PATTriggerMatcher.cc,v 1.1.2.1 2009/03/15 12:24:15 vadler Exp $
//
#include "PhysicsTools/PatAlgos/plugins/PATTriggerMatchSelector.h"
#include "PhysicsTools/UtilAlgos/interface/PhysObjectMatcher.h"
#include "PhysicsTools/UtilAlgos/interface/MatchByDR.h"
#include "PhysicsTools/UtilAlgos/interface/MatchByDRDPt.h"
#include "PhysicsTools/UtilAlgos/interface/MatchLessByDPt.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/TriggerObject.h"


/// Match by deltaR (default), ranking by deltaR (default)
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  pat::TriggerObjectCollection,
  pat::PATTriggerMatchSelector< reco::CandidateView::value_type,
                                pat::TriggerObjectCollection::value_type >
> PATTriggerMatcherDRLessByR;

/// Match by deltaR and deltaPt, ranking by deltaR (default)
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  pat::TriggerObjectCollection,
  pat::PATTriggerMatchSelector< reco::CandidateView::value_type,
                                pat::TriggerObjectCollection::value_type>,
  reco::MatchByDRDPt< reco::CandidateView::value_type,
                      pat::TriggerObjectCollection::value_type >
> PATTriggerMatcherDRDPtLessByR;

/// Match by deltaR (default), ranking by deltaPt
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  pat::TriggerObjectCollection,
  pat::PATTriggerMatchSelector< reco::CandidateView::value_type,
                                pat::TriggerObjectCollection::value_type >,
  reco::MatchByDR< reco::CandidateView::value_type,
                   pat::TriggerObjectCollection::value_type >,
  reco::MatchLessByDPt< reco::CandidateView,
                        pat::TriggerObjectCollection >
> PATTriggerMatcherDRLessByPt;

/// Match by deltaR and deltaPt, ranking by deltaPt
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  pat::TriggerObjectCollection,
  pat::PATTriggerMatchSelector<reco::CandidateView::value_type,
                               pat::TriggerObjectCollection::value_type >,
  reco::MatchByDRDPt< reco::CandidateView::value_type,
                      pat::TriggerObjectCollection::value_type  >,
  reco::MatchLessByDPt< reco::CandidateView,
                        pat::TriggerObjectCollection >
> PATTriggerMatcherDRDPtLessByPt;


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PATTriggerMatcherDRLessByR );
DEFINE_FWK_MODULE( PATTriggerMatcherDRDPtLessByR );
DEFINE_FWK_MODULE( PATTriggerMatcherDRLessByPt );
DEFINE_FWK_MODULE( PATTriggerMatcherDRDPtLessByPt );

