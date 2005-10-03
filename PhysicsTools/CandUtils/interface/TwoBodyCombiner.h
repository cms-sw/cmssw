#ifndef PHYSICSTOOLS_TWOBODYCOMBINER_H
#define PHYSICSTOOLS_TWOBODYCOMBINER_H
// $Id: TwoBodyCombiner.h,v 1.3 2005/10/03 08:16:48 llista Exp $
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/Candidate/interface/OverlapChecker.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"

class TwoBodyCombiner {
public:
  typedef phystools::Candidate::collection Candidates;
  TwoBodyCombiner( double massMin, double massMax, 
		   bool checkCharge, int charge = 0 );
protected:
  bool select( const phystools::Candidate &, const phystools::Candidate & ) const;
  phystools::Candidate * combine( const phystools::Candidate &, const phystools::Candidate & );

  double mass2min, mass2max;
  bool checkCharge;
  int charge;
  phystools::AddFourMomenta addp4;
  phystools::OverlapChecker overlap;
};

#endif
