#ifndef PHYSICSTOOLS_TWOBODYCOMBINER_H
#define PHYSICSTOOLS_TWOBODYCOMBINER_H
// $Id: TwoBodyCombiner.h,v 1.1 2005/10/03 09:17:45 llista Exp $
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/Candidate/interface/OverlapChecker.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"

class TwoBodyCombiner {
public:
  typedef aod::Candidate::collection Candidates;
  TwoBodyCombiner( double massMin, double massMax, 
		   bool checkCharge, int charge = 0 );
protected:
  bool select( const aod::Candidate &, const aod::Candidate & ) const;
  aod::Candidate * combine( const aod::Candidate &, const aod::Candidate & );

  double mass2min, mass2max;
  bool checkCharge;
  int charge;
  AddFourMomenta addp4;
  OverlapChecker overlap;
};

#endif
