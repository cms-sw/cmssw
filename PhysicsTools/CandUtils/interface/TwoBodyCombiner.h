#ifndef PHYSICSTOOLS_TWOBODYCOMBINER_H
#define PHYSICSTOOLS_TWOBODYCOMBINER_H
// $Id: TwoBodyCombiner.h,v 1.3 2005/10/24 10:14:25 llista Exp $
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/Selector.h"
#include "PhysicsTools/Candidate/interface/OverlapChecker.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include <boost/shared_ptr.hpp>

class TwoBodyCombiner {
public:
  typedef aod::Candidate::collection Candidates;
  TwoBodyCombiner( const boost::shared_ptr<aod::Selector> &, 
		   bool checkCharge, int charge = 0 );
  std::auto_ptr<Candidates> combine( const Candidates *, const Candidates * );
protected:
  bool preselect( const aod::Candidate &, const aod::Candidate & ) const;
 aod::Candidate * combine( const aod::Candidate &, const aod::Candidate & );

  bool checkCharge;
  int charge;
  AddFourMomenta addp4;
  OverlapChecker overlap;
  boost::shared_ptr<aod::Selector> select;
};

#endif
