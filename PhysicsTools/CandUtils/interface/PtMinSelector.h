#ifndef PHYSICSTOOLS_PTMINSELECTOR_H
#define PHYSICSTOOLS_PTMINSELECTOR_H
// $Id: PtMinSelector.h,v 1.6 2005/10/25 08:47:05 llista Exp $
#include "PhysicsTools/CandUtils/interface/CandSelector.h"

class PtMinSelector : public CandSelector {
public:
  explicit PtMinSelector( double cut ) :
    ptMin( cut ) {
  }
  bool operator()( const aod::Candidate & c ) const;
private:
  double ptMin;
};

#endif
