#ifndef CandUtils_TwoBodyCombiner_h
#define CandUtils_TwoBodyCombiner_h
// $Id: TwoBodyCombiner.h,v 1.8 2006/02/21 10:37:30 llista Exp $
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include <boost/shared_ptr.hpp>

class TwoBodyCombiner {
public:
  TwoBodyCombiner( const boost::shared_ptr<CandSelector> &, 
		   bool checkCharge, int charge = 0 );
  std::auto_ptr<reco::CandidateCollection> 
  combine( const reco::CandidateCollection *, const reco::CandidateCollection * );
protected:
  bool preselect( const reco::Candidate &, const reco::Candidate & ) const;
 reco::Candidate * combine( const reco::Candidate &, const reco::Candidate & );

  bool checkCharge;
  int charge;
  AddFourMomenta addp4;
  OverlapChecker overlap;
  boost::shared_ptr<CandSelector> select;
};

#endif
