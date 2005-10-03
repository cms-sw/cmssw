#ifndef PHYSICSTOOLS_MASSWINDOWSELECTOR_H
#define PHYSICSTOOLS_MASSWINDIWSELECTOR_H
// $Id: MassWindowSelector.h,v 1.2 2005/10/03 09:17:45 llista Exp $
#include "PhysicsTools/Candidate/interface/Candidate.h"

class MassWindowSelector {
public:
  explicit MassWindowSelector( const edm::ParameterSet & parms ) :
    mMin2( parms.getParameter<double>( "massMin" ) ),
    mMax2( parms.getParameter<double>( "massMax" ) ) {
    mMin2 *= mMin2;
    mMax2 *= mMax2;
  }
  bool operator()( const aod::Candidate * c ) const {
    double m2 = c->massSqr();
    return mMin2 < m2 && m2 < mMax2;
  }
private:
  double mMin2, mMax2;
};

#endif
