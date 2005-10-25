#ifndef PHYSICSTOOLS_MASSWINDOWSELECTOR_H
#define PHYSICSTOOLS_MASSWINDIWSELECTOR_H
// $Id: MassWindowSelector.h,v 1.5 2005/10/21 13:56:43 llista Exp $
#include "PhysicsTools/Candidate/interface/Candidate.h"

class MassWindowSelector : public aod::Candidate::selector {
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
