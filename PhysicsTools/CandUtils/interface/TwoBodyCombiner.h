#ifndef PHYSICSTOOLS_TWOBODYCOMBINER_H
#define PHYSICSTOOLS_TWOBODYCOMBINER_H
// $Id: TwoBodyCombiner.h,v 1.2 2005/10/03 10:12:11 llista Exp $
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/Candidate/interface/OverlapChecker.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"

class TwoBodyCombiner {
public:
  typedef aod::Candidate::collection Candidates;
  TwoBodyCombiner( double massMin, double massMax, 
		   bool checkCharge, int charge = 0 );
  std::auto_ptr<Candidates> combine( const Candidates *, const Candidates * );
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
