#ifndef PHYSICSTOOLS_MASSWINDOWSELECTOR_H
#define PHYSICSTOOLS_MASSWINDIWSELECTOR_H
// $Id: MassWindowSelector.h,v 1.3 2005/10/03 10:12:11 llista Exp $
#include "PhysicsTools/CandUtils/interface/Selector.h"

class MassWindowSelector : public aod::Selector {
public:
  explicit MassWindowSelector( double massMin, double massMax ) :
    mMin2( massMin ), mMax2( massMax ) {
    mMin2 *= mMin2;
    mMax2 *= mMax2;
  }
  bool operator()( const aod::Candidate * c ) const;
private:
  double mMin2, mMax2;
};

#endif
