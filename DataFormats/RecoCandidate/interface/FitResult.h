#ifndef RecoCandidate_FitResult_h
#define RecoCandidate_FitResult_h

#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/FitQuality.h"

namespace reco {
  typedef edm::AssociationVector<CandidateRefProd, std::vector<FitQuality> > FitResultCollection;
}

#endif
