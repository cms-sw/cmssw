#ifndef CandUtils_NBodyCombiner_h
#define CandUtils_NBodyCombiner_h
/** \class NBodyCombiner
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include <vector>
#include <utility>

class NBodyCombinerBase {
public:
  /// constructor from a selector, specifying optionally to check for charge
  NBodyCombinerBase( bool checkCharge, const std::vector <int> & );
  /// destructor
  virtual ~NBodyCombinerBase();
  /// return all selected candidate pairs
  std::auto_ptr<reco::CandidateCollection> 
  combine( const std::vector<reco::CandidateRefProd> & ) const;

private:
  /// verify that the two candidate don't overlap and check charge
  bool preselect( const reco::Candidate &, const reco::Candidate & ) const;
  /// returns a composite candidate combined from two daughters
  reco::Candidate * combine( const reco::CandidateRef &, const reco::CandidateRef & ) const;
  /// charge information flag
  enum ChargeInfo { undetermined, same, opposite, invalid };
  /// return charge information
  static ChargeInfo chargeInfo( int q1, int q2 ); 
  /// temporary candidate stack
  typedef std::vector<std::pair<reco::CandidateRef, 
                                std::vector<reco::CandidateRefProd>::const_iterator> 
                     > CandStack;
  /// returns a composite candidate combined from two daughters
  void combine( size_t collectionIndex, ChargeInfo ch, CandStack &, 
		std::vector<reco::CandidateRefProd>::const_iterator begin,
		std::vector<reco::CandidateRefProd>::const_iterator end,
		std::auto_ptr<reco::CandidateCollection> & comps
		) const;
  /// select a candidate
  virtual bool select( const reco::Candidate & ) const = 0;
  /// set kinematics to reconstructed composite
  virtual void setup( reco::Candidate * ) const = 0;
  /// add candidate daughter
  virtual void addDaughter( reco::CompositeCandidate * cmp, const reco::CandidateRef & c ) const = 0;
  /// flag to specify the checking of electric charge
  bool checkCharge_;
  /// electric charges of the daughters
  std::vector<int> dauCharge_;
  /// utility to check candidate daughters overlap
  OverlapChecker overlap_;
};

namespace combiner {
  namespace helpers {
    struct NormalClone {
      static void addDaughter( reco::CompositeCandidate & cmp, const reco::CandidateRef & c ) {
	cmp.addDaughter( * c );
      }
    };
    
    struct ShallowClone {
      static void addDaughter( reco::CompositeCandidate & cmp, const reco::CandidateRef & c ) {
	cmp.addDaughter( reco::ShallowCloneCandidate( reco::CandidateBaseRef( c ) ) );
      }
    };
  }
}

template<typename S, typename H = combiner::helpers::NormalClone, typename Setup = AddFourMomenta>
class NBodyCombiner : public NBodyCombinerBase {
public:
  /// constructor from a selector, specifying optionally to check for charge
  template<typename B>
  NBodyCombiner( const B & select,
		 bool checkCharge, const std::vector <int> & dauCharge ) : 
    NBodyCombinerBase( checkCharge, dauCharge ), 
    select_( select ), setup_() { }
  /// constructor from a selector, specifying optionally to check for charge
  NBodyCombiner( const edm::ParameterSet & cfg,
		 bool checkCharge, const std::vector <int> & dauCharge ) : 
    NBodyCombinerBase( checkCharge, dauCharge ), 
    select_( cfg ), setup_( cfg ) { }
  /// return reference to setup object to allow its initialization
  Setup & setup() { return setup_; }
private:
  /// select a candidate
  virtual bool select( const reco::Candidate & c ) const {
    return select_( c );
  } 
  /// set kinematics to reconstructed composite
  virtual void setup( reco::Candidate * c ) const {
    setup_.set( * c );
  }
  /// add candidate daughter
  virtual void addDaughter( reco::CompositeCandidate * cmp, const reco::CandidateRef & c ) const {
    H::addDaughter( * cmp, c );
  }
  /// candidate selector
  S select_; 
  /// utility to setup composite candidate kinematics from daughters
  Setup setup_;
};

#endif
