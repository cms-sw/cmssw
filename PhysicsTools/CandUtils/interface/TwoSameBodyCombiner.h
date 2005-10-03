#ifndef PHYSICSTOOLS_TWOSAMEBODYCOMBINER_H
#define PHYSICSTOOLS_TWOSAMEBODYCOMBINER_H
// $Id: TwoSameBodyCombiner.h,v 1.1 2005/07/29 07:22:52 llista Exp $
#include "PhysicsTools/CandUtils/interface/TwoBodyCombiner.h"
#include <memory>
  
class TwoSameBodyCombiner : public TwoBodyCombiner {
public:
  TwoSameBodyCombiner( double massMin, double massMax, 
		       bool checkCharge, int charge = 0 );
  std::auto_ptr<Candidates> combine( const Candidates & );
};

#endif
