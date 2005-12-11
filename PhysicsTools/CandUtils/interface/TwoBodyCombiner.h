#ifndef CandUtils_TwoBodyCombiner_h
#define CandUtils_TwoBodyCombiner_h
// $Id: TwoBodyCombiner.h,v 1.6 2005/10/25 09:08:31 llista Exp $
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/Candidate/interface/OverlapChecker.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include <boost/shared_ptr.hpp>

class TwoBodyCombiner {
public:
  TwoBodyCombiner( const boost::shared_ptr<CandSelector> &, 
		   bool checkCharge, int charge = 0 );
  std::auto_ptr<aod::CandidateCollection> 
  combine( const aod::CandidateCollection *, const aod::CandidateCollection * );
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
