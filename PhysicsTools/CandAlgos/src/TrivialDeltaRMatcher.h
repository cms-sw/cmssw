#ifndef TrivialDeltaRMatcher_h
#define TrivialDeltaRMatcher_h
/* \class DeltaRMatcher
 *
 * Producer fo simple match map
 * based on DeltaR
 *
 */
#include "PhysicsTools/CandAlgos/interface/CandMatcher.h"
#include "PhysicsTools/Utilities/interface/AnySelector.h"

typedef CandMatcher<AnySelector<reco::Candidate> > TrivialDeltaRMatcher;

#endif
