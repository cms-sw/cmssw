#ifndef RecoTauTag_TauTagTools_PFTauDecayModeTruthMatcher
#define RecoTauTag_TauTagTools_PFTauDecayModeTruthMatcher

#include "CommonTools/UtilAlgos/interface/PhysObjectMatcher.h"
#include "CommonTools/UtilAlgos/interface/MCMatchSelector.h"
#include "CommonTools/UtilAlgos/interface/DummyMatchSelector.h"
#include "CommonTools/UtilAlgos/interface/MatchByDRDPt.h"
#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"

// match by tau candidates to MC by DR/PT, ranking by deltaR
// Author: Evan Friis, UC Davis, evan.klose.friis@cern.ch

typedef reco::PhysObjectMatcher<
                reco::PFTauDecayModeCollection,
//                reco::PFTauDecayModeCollection,
                reco::PFTauCollection,
                reco::DummyMatchSelector<reco::PFTauDecayModeCollection::value_type,
//                                      reco::PFTauDecayModeCollection::value_type>,
                                      reco::PFTauCollection::value_type>,
                reco::MatchByDRDPt<reco::PFTauDecayModeCollection::value_type,
//                                   reco::PFTauDecayModeCollection::value_type>
                                      reco::PFTauCollection::value_type>
        > PFTauDecayModeTruthMatcher;

#endif
