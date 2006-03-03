#ifndef CandUtils_TwoBodyCombiner_h
#define CandUtils_TwoBodyCombiner_h
/** \class TwoBodyCombiner
 *
 * Performs all possible combination of candidate pairs,
 * selects the combinations via a specified CandSelector,
 * with the possibility to check the composite candidate
 * electric charge. The algorithm also checks that the
 * paired candidates do not overlap.
 * 
 * If the same input collection is passed twice, the 
 * algorithm avoids double counting the candidate pairs
 *
 * The composite candidate kinematics is set up 
 * adding the two daughters four-momenta
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision$
 *
 * $Id: Track.h,v 1.12 2006/03/01 12:23:40 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include <boost/shared_ptr.hpp>

class TwoBodyCombiner {
public:
  /// constructor from a selector, specifying optionally to check for charge
  TwoBodyCombiner( const boost::shared_ptr<CandSelector> &, 
		   bool checkCharge, int charge = 0 );
  /// return all selected candidate pairs
  std::auto_ptr<reco::CandidateCollection> 
  combine( const reco::CandidateCollection *, const reco::CandidateCollection * );
protected:
  /// verify that the two candidate don't overlap and check charge
  bool preselect( const reco::Candidate &, const reco::Candidate & ) const;
  /// returns a composite candidate combined from two daughters
  reco::Candidate * combine( const reco::Candidate &, const reco::Candidate & );
  /// flag to specify the checking of electric charge
  bool checkCharge;
  /// electric charge of the composite
  int charge;
  /// utility to setup composite candidate kinematics from daughters
  AddFourMomenta addp4;
  /// utility to check candidate daughters overlap
  OverlapChecker overlap;
  /// candidate selector
  boost::shared_ptr<CandSelector> select;
};

#endif
