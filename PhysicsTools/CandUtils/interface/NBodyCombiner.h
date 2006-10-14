#ifndef CandUtils_NBodyCombiner_h
#define CandUtils_NBodyCombiner_h
/** \class ThreeBodyCombiner
 *
 */
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "PhysicsTools/Parser/interface/SingleObjectSelector.h"
#include <vector>

class NBodyCombiner {
public:
  /// constructor from a selector, specifying optionally to check for charge
  NBodyCombiner( const reco::parser::SelectorPtr &, 
		 bool checkCharge, const std::vector <int> & );
  /// return all selected candidate pairs
  std::auto_ptr<reco::CandidateCollection> 
    combine( const std::vector<const reco::CandidateCollection *> & ) const;
protected:
  /// verify that the two candidate don't overlap and check charge
  bool preselect( const reco::Candidate &, const reco::Candidate & ) const;
  /// returns a composite candidate combined from two daughters
  reco::Candidate * combine( const reco::Candidate &, const reco::Candidate & ) const;
  /// charge information flag
  enum ChargeInfo { undetermined, same, opposite, invalid };
  /// return charge information
  static ChargeInfo chargeInfo( int q1, int q2 ); 
  /// returns a composite candidate combined from two daughters
  void combine( size_t collectionIndex, ChargeInfo ch, std::vector<const reco::Candidate *> cv,
		const std::vector<const reco::CandidateCollection * >::const_iterator begin,
		const std::vector<const reco::CandidateCollection * >::const_iterator end,
		std::auto_ptr<reco::CandidateCollection> & comps
		) const;
  /// flag to specify the checking of electric charge
  bool checkCharge_;
  /// electric charges of the daughters
  std::vector<int> dauCharge_;
  /// utility to setup composite candidate kinematics from daughters
  AddFourMomenta addp4_;
  /// utility to check candidate daughters overlap
  OverlapChecker overlap_;
  /// candidate selector
  SingleObjectSelector<reco::Candidate> select_;
};

#endif
