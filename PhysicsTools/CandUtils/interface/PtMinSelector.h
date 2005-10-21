#ifndef PHYSICSTOOLS_PTMINSELECTOR_H
#define PHYSICSTOOLS_PTMINSELECTOR_H
// $Id: PtMinSelector.h,v 1.4 2005/10/21 12:44:35 llista Exp $
#include "PhysicsTools/CandUtils/interface/Selector.h"

class PtMinSelector : public aod::Selector {
public:
  explicit PtMinSelector( double cut ) :
    ptMin( cut ) {
  }
  bool operator()( const aod::Candidate & c ) const;
private:
  double ptMin;
};

#endif
