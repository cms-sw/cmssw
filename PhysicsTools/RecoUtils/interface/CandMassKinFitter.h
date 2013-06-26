#ifndef RecoCandUtils_CandMassKinFitter_h
#define RecoCandUtils_CandMassKinFitter_h
#include "DataFormats/RecoCandidate/interface/FitQuality.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

struct CandMassKinFitter {
  explicit CandMassKinFitter( double m ) : mass_( m ) { }
  reco::FitQuality set( reco::Candidate & ) const;
private:
  double mass_;
};

#endif
