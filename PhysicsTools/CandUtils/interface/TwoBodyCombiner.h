#ifndef CandUtils_TwoBodyCombiner_h
#define CandUtils_TwoBodyCombiner_h
// $Id: TwoBodyCombiner.h,v 1.5 2005/10/25 08:47:05 llista Exp $
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/Candidate/interface/OverlapChecker.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include <boost/shared_ptr.hpp>

class TwoBodyCombiner {
public:
  typedef aod::Candidate::collection Candidates;
  TwoBodyCombiner( const boost::shared_ptr<CandSelector> &, 
		   bool checkCharge, int charge = 0 );
  std::auto_ptr<Candidates> combine( const Candidates *, const Candidates * );
protected:
  bool preselect( const aod::Candidate &, const aod::Candidate & ) const;
 aod::Candidate * combine( const aod::Candidate &, const aod::Candidate & );

  bool checkCharge;
  int charge;
  AddFourMomenta addp4;
  OverlapChecker overlap;
  boost::shared_ptr<CandSelector> select;
};

#endif
