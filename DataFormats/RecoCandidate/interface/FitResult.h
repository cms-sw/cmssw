#ifndef RecoCandidate_FitResult_h
#define RecoCandidate_FitResult_h

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/FitQuality.h"

namespace reco {
  typedef edm::ValueMap<FitQuality> FitResultCollection;
}

#endif
