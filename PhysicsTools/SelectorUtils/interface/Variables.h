#ifndef SelectorUtils_Variables_h
#define SelectorUtils_Variables_h

// short cut classes for reco::Candidate methods
// Benedikt Hegner, DESY

#include "DataFormats/Candidate/interface/Candidate.h"

class vEnergy {
public:
  typedef reco::Candidate ValType;
  vEnergy() {}
  double operator()( const ValType& x ) const { return x.energy(); }
};

#endif
