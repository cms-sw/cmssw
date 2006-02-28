#ifndef CandUtils_AddFourMomenta_h
#define CandUtils_AddFourMomenta_h
// $Id: AddFourMomenta.h,v 1.3 2006/02/21 10:37:30 llista Exp $
#include "DataFormats/Candidate/interface/Candidate.h"

struct AddFourMomenta : public reco::Candidate::setup {
  AddFourMomenta() : reco::Candidate::setup( setupCharge( true ), setupP4( true ) ) { }
  virtual ~AddFourMomenta();
  void set( reco::Candidate& c );
};

#endif
