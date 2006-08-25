#ifndef CandUtils_ThreeBodyCombiner_h
#define CandUtils_ThreeBodyCombiner_h
/** \class ThreeBodyCombiner
 *
 */
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "PhysicsTools/Parser/interface/SingleObjectSelector.h"
#include <vector>

class ThreeBodyCombiner {
public:
  /// constructor from a selector, specifying optionally to check for charge
  ThreeBodyCombiner( const reco::parser::SelectorPtr &, 
		     bool checkCharge, const std::vector <int> &,
		     int charge = 0);
  /// return all selected candidate pairs
  std::auto_ptr<reco::CandidateCollection> 
    combine( const reco::CandidateCollection *, const reco::CandidateCollection *,
	     const reco::CandidateCollection * );
  void combineWithTwoEqualCollection( const reco::CandidateCollection *, 
				      const reco::CandidateCollection *,
				      std::auto_ptr<reco::CandidateCollection> );
protected:
  /// verify that the two candidate don't overlap and check charge
  bool preselect( const reco::Candidate &, const reco::Candidate &,
		  const reco::Candidate & ) const;
  /// returns a composite candidate combined from two daughters
  reco::Candidate * combine( const reco::Candidate &, const reco::Candidate &,
			     const reco::Candidate & );
  /// flag to specify the checking of electric charge
  bool checkCharge_;
  /// electric charges of the daughters
  std::vector<int> dauCharge_;
  /// electric charge of the composite
  int charge_;
  /// utility to setup composite candidate kinematics from daughters
  AddFourMomenta addp4_;
  /// utility to check candidate daughters overlap
  OverlapChecker overlap_;
  /// candidate selector
  SingleObjectSelector<reco::Candidate> select_;
};

#endif
