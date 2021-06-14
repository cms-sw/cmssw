// -*- C++ -*-
//
// Package:    PhysicsTools/PatAlgos
// Class:      PATTriggerMatchSelector
//
/**
  \class    pat::PATTriggerMatchSelector PATTriggerMatchSelector.cc "PhysicsTools/PatAlgos/plugins/PATTriggerMatchSelector.cc"
  \brief

   .

  \author   Volker Adler
  \version  $Id: PATTriggerMatchSelector.h,v 1.5 2010/06/16 15:40:58 vadler Exp $
*/

#include "CommonTools/UtilAlgos/interface/MatchByDEta.h"
#include "CommonTools/UtilAlgos/interface/MatchByDR.h"
#include "CommonTools/UtilAlgos/interface/MatchByDRDPt.h"
#include "CommonTools/UtilAlgos/interface/MatchLessByDEta.h"
#include "CommonTools/UtilAlgos/interface/MatchLessByDPt.h"
#include "CommonTools/UtilAlgos/interface/PhysObjectMatcher.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <map>
#include <string>
#include <vector>

namespace pat {

  template <typename T1, typename T2>
  class PATTriggerMatchSelector : public StringCutObjectSelector<T2> {
  public:
    PATTriggerMatchSelector(const edm::ParameterSet& iConfig)
        : StringCutObjectSelector<T2>(iConfig.getParameter<std::string>("matchedCuts")) {}

    bool operator()(const T1& patObj, const T2& trigObj) const {
      return StringCutObjectSelector<T2>::operator()(trigObj);
    }
  };

}  // namespace pat

/// Match by deltaR (default), ranking by deltaR (default)
typedef reco::PhysObjectMatcher<
    reco::CandidateView,
    pat::TriggerObjectStandAloneCollection,
    pat::PATTriggerMatchSelector<reco::CandidateView::value_type, pat::TriggerObjectStandAloneCollection::value_type> >
    PATTriggerMatcherDRLessByR;

/// Match by deltaR and deltaPt, ranking by deltaR (default)
typedef reco::PhysObjectMatcher<
    reco::CandidateView,
    pat::TriggerObjectStandAloneCollection,
    pat::PATTriggerMatchSelector<reco::CandidateView::value_type, pat::TriggerObjectStandAloneCollection::value_type>,
    reco::MatchByDRDPt<reco::CandidateView::value_type, pat::TriggerObjectStandAloneCollection::value_type> >
    PATTriggerMatcherDRDPtLessByR;

/// Match by deltaR (default), ranking by deltaPt
typedef reco::PhysObjectMatcher<
    reco::CandidateView,
    pat::TriggerObjectStandAloneCollection,
    pat::PATTriggerMatchSelector<reco::CandidateView::value_type, pat::TriggerObjectStandAloneCollection::value_type>,
    reco::MatchByDR<reco::CandidateView::value_type, pat::TriggerObjectStandAloneCollection::value_type>,
    reco::MatchLessByDPt<reco::CandidateView, pat::TriggerObjectStandAloneCollection> >
    PATTriggerMatcherDRLessByPt;

/// Match by deltaR and deltaPt, ranking by deltaPt
typedef reco::PhysObjectMatcher<
    reco::CandidateView,
    pat::TriggerObjectStandAloneCollection,
    pat::PATTriggerMatchSelector<reco::CandidateView::value_type, pat::TriggerObjectStandAloneCollection::value_type>,
    reco::MatchByDRDPt<reco::CandidateView::value_type, pat::TriggerObjectStandAloneCollection::value_type>,
    reco::MatchLessByDPt<reco::CandidateView, pat::TriggerObjectStandAloneCollection> >
    PATTriggerMatcherDRDPtLessByPt;

/// Match by deltaEta, ranking by deltaR
typedef reco::PhysObjectMatcher<
    reco::CandidateView,
    pat::TriggerObjectStandAloneCollection,
    pat::PATTriggerMatchSelector<reco::CandidateView::value_type, pat::TriggerObjectStandAloneCollection::value_type>,
    reco::MatchByDEta<reco::CandidateView::value_type, pat::TriggerObjectStandAloneCollection::value_type> >
    PATTriggerMatcherDEtaLessByDR;

/// Match by deltaEta, ranking by deltaEta
typedef reco::PhysObjectMatcher<
    reco::CandidateView,
    pat::TriggerObjectStandAloneCollection,
    pat::PATTriggerMatchSelector<reco::CandidateView::value_type, pat::TriggerObjectStandAloneCollection::value_type>,
    reco::MatchByDEta<reco::CandidateView::value_type, pat::TriggerObjectStandAloneCollection::value_type>,
    reco::MatchLessByDEta<reco::CandidateView, pat::TriggerObjectStandAloneCollection> >
    PATTriggerMatcherDEtaLessByDEta;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATTriggerMatcherDRLessByR);
DEFINE_FWK_MODULE(PATTriggerMatcherDRDPtLessByR);
DEFINE_FWK_MODULE(PATTriggerMatcherDRLessByPt);
DEFINE_FWK_MODULE(PATTriggerMatcherDRDPtLessByPt);
DEFINE_FWK_MODULE(PATTriggerMatcherDEtaLessByDR);
DEFINE_FWK_MODULE(PATTriggerMatcherDEtaLessByDEta);
