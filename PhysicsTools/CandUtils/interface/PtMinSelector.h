#ifndef PHYSICSTOOLS_PTMINSELECTOR_H
#define PHYSICSTOOLS_PTMINSELECTOR_H
// $Id: PtMinSelector.h,v 1.5 2005/10/21 13:56:43 llista Exp $
#include "PhysicsTools/Candidate/interface/Candidate.h"

class PtMinSelector : public aod::Candidate::selector {
public:
  explicit PtMinSelector( double cut ) :
    ptMin( cut ) {
  }
  bool operator()( const aod::Candidate & c ) const;
private:
  double ptMin;
};

#endif
