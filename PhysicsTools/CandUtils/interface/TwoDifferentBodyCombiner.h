#ifndef PHYSICSTOOLS_TWODIFFERENTBODYCOMBINER_H
#define PHYSICSTOOLS_TWODIFFERENTBODYCOMBINER_H
// $Id: TwoDifferentBodyCombiner.h,v 1.1 2005/07/29 07:22:52 llista Exp $
#include "PhysicsTools/CandUtils/interface/TwoBodyCombiner.h"
#include <memory>

class TwoDifferentBodyCombiner : public TwoBodyCombiner {
public:
  TwoDifferentBodyCombiner( double massMin, double massMax, 
			    bool checkCharge, int charge = 0 );
  std::auto_ptr<Candidates> combine( const Candidates &, const Candidates & );
};

#endif
