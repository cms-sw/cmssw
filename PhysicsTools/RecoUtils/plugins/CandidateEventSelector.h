#include "CommonTools/UtilAlgos/interface/StringCutEventSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef StringCutEventSelector<reco::Candidate> CandidateEventSelector;
typedef StringCutsEventSelector<reco::Candidate> CandidateSEventSelector;
typedef StringCutsEventSelector<reco::Candidate,false> CandidateSEventVetoSelector;
