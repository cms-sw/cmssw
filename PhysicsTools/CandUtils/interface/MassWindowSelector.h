#ifndef PHYSICSTOOLS_MASSWINDOWSELECTOR_H
#define PHYSICSTOOLS_MASSWINDIWSELECTOR_H
// $Id: MassWindowSelector.h,v 1.6 2005/10/25 08:47:05 llista Exp $
#include "PhysicsTools/CandUtils/interface/CandSelector.h"

class MassWindowSelector : public CandSelector {
public:
  explicit MassWindowSelector( double massMin, double massMax ) :
    mMin2( massMin ), mMax2( massMax ) {
    mMin2 *= mMin2;
    mMax2 *= mMax2;
  }
  bool operator()( const aod::Candidate & c ) const;
private:
  double mMin2, mMax2;
};

#endif
