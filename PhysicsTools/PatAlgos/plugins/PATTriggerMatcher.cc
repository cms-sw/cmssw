//
//
#include "PhysicsTools/PatAlgos/plugins/PATTriggerMatchSelector.h"
#include "CommonTools/UtilAlgos/interface/PhysObjectMatcher.h"
#include "CommonTools/UtilAlgos/interface/MatchByDR.h"
#include "CommonTools/UtilAlgos/interface/MatchByDRDPt.h"
#include "CommonTools/UtilAlgos/interface/MatchLessByDPt.h"
#include "CommonTools/UtilAlgos/interface/MatchByDEta.h"
#include "CommonTools/UtilAlgos/interface/MatchLessByDEta.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"


/// Match by deltaR (default), ranking by deltaR (default)
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  pat::TriggerObjectStandAloneCollection,
  pat::PATTriggerMatchSelector< reco::CandidateView::value_type,
                                pat::TriggerObjectStandAloneCollection::value_type >
> PATTriggerMatcherDRLessByR;

/// Match by deltaR and deltaPt, ranking by deltaR (default)
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  pat::TriggerObjectStandAloneCollection,
  pat::PATTriggerMatchSelector< reco::CandidateView::value_type,
                                pat::TriggerObjectStandAloneCollection::value_type>,
  reco::MatchByDRDPt< reco::CandidateView::value_type,
                      pat::TriggerObjectStandAloneCollection::value_type >
> PATTriggerMatcherDRDPtLessByR;

/// Match by deltaR (default), ranking by deltaPt
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  pat::TriggerObjectStandAloneCollection,
  pat::PATTriggerMatchSelector< reco::CandidateView::value_type,
                                pat::TriggerObjectStandAloneCollection::value_type >,
  reco::MatchByDR< reco::CandidateView::value_type,
                   pat::TriggerObjectStandAloneCollection::value_type >,
  reco::MatchLessByDPt< reco::CandidateView,
                        pat::TriggerObjectStandAloneCollection >
> PATTriggerMatcherDRLessByPt;

/// Match by deltaR and deltaPt, ranking by deltaPt
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  pat::TriggerObjectStandAloneCollection,
  pat::PATTriggerMatchSelector<reco::CandidateView::value_type,
                               pat::TriggerObjectStandAloneCollection::value_type >,
  reco::MatchByDRDPt< reco::CandidateView::value_type,
                      pat::TriggerObjectStandAloneCollection::value_type >,
  reco::MatchLessByDPt< reco::CandidateView,
                        pat::TriggerObjectStandAloneCollection >
> PATTriggerMatcherDRDPtLessByPt;

/// Match by deltaEta, ranking by deltaR
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  pat::TriggerObjectStandAloneCollection,
  pat::PATTriggerMatchSelector< reco::CandidateView::value_type,
                                pat::TriggerObjectStandAloneCollection::value_type >,
  reco::MatchByDEta< reco::CandidateView::value_type,
                     pat::TriggerObjectStandAloneCollection::value_type >
> PATTriggerMatcherDEtaLessByDR;

/// Match by deltaEta, ranking by deltaEta
typedef reco::PhysObjectMatcher<
  reco::CandidateView,
  pat::TriggerObjectStandAloneCollection,
  pat::PATTriggerMatchSelector< reco::CandidateView::value_type,
                                pat::TriggerObjectStandAloneCollection::value_type >,
  reco::MatchByDEta< reco::CandidateView::value_type,
                     pat::TriggerObjectStandAloneCollection::value_type >,
  reco::MatchLessByDEta< reco::CandidateView,
                         pat::TriggerObjectStandAloneCollection >
> PATTriggerMatcherDEtaLessByDEta;


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE( PATTriggerMatcherDRLessByR );
DEFINE_FWK_MODULE( PATTriggerMatcherDRDPtLessByR );
DEFINE_FWK_MODULE( PATTriggerMatcherDRLessByPt );
DEFINE_FWK_MODULE( PATTriggerMatcherDRDPtLessByPt );
DEFINE_FWK_MODULE( PATTriggerMatcherDEtaLessByDR );
DEFINE_FWK_MODULE( PATTriggerMatcherDEtaLessByDEta );
