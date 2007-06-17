#ifndef CandUtils_CandCombiner_h
#define CandUtils_CandCombiner_h
/** \class CandCombiner
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include <vector>
#include <utility>

class CandCombinerBase {
public:
  /// default construct
  CandCombinerBase();
  /// construct from two charge values
  CandCombinerBase( int, int );
  /// construct from three charge values
  CandCombinerBase( int, int, int );
  /// construct from four charge values
  CandCombinerBase( int, int, int, int );
  /// constructor from a selector, specifying optionally to check for charge
  CandCombinerBase( bool checkCharge, const std::vector <int> & );
  /// destructor
  virtual ~CandCombinerBase();
  /// return all selected candidate pairs
  std::auto_ptr<reco::CandidateCollection> 
  combine( const std::vector<reco::CandidateRefProd> & ) const;
  /// return all selected candidate pairs
  std::auto_ptr<reco::CandidateCollection> 
  combine( const reco::CandidateRefProd & ) const;

private:
  /// verify that the two candidate don't overlap and check charge
  bool preselect( const reco::Candidate &, const reco::Candidate & ) const;
  /// returns a composite candidate combined from two daughters
  reco::Candidate * combine( const reco::CandidateRef &, const reco::CandidateRef & ) const;
  /// temporary candidate stack
  typedef std::vector<std::pair<reco::CandidateRef, 
                      std::vector<reco::CandidateRefProd>::const_iterator> 
                     > CandStack;
  typedef std::vector<int> ChargeStack;
  /// returns a composite candidate combined from two daughters
  void combine( size_t collectionIndex, CandStack &, ChargeStack &,
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
class CandCombiner : public CandCombinerBase {
public:
  /// default constructor
  CandCombiner() :
    CandCombinerBase( ), 
    select_(), setup_() { }
  /// constructor from a selector and two charges
  CandCombiner( int q1, int q2 ) :
    CandCombinerBase( q1, q2 ), 
    select_(), setup_() { }
  /// constructor from a selector and three charges
  CandCombiner( int q1, int q2, int q3 ) :
    CandCombinerBase( q1, q2, q3 ), 
    select_(), setup_() { }
  /// constructor from a selector and four charges
  CandCombiner( int q1, int q2, int q3, int q4 ) :
    CandCombinerBase( q1, q2, q3, q4 ), 
    select_(), setup_() { }
  /// default constructor
  CandCombiner( const S & select ) :
    CandCombinerBase( ), 
    select_( select ), setup_() { }
  /// constructor from a selector and two charges
  CandCombiner( const S & select, int q1, int q2 ) :
    CandCombinerBase( q1, q2 ), 
    select_( select ), setup_() { }
  /// constructor from a selector and three charges
  CandCombiner( const S & select, int q1, int q2, int q3 ) :
    CandCombinerBase( q1, q2, q3 ), 
    select_( select ), setup_() { }
  /// constructor from a selector and four charges
  CandCombiner( const S & select, int q1, int q2, int q3, int q4 ) :
    CandCombinerBase( q1, q2, q3, q4 ), 
    select_( select ), setup_() { }
  CandCombiner( const S & select, const Setup & setup ) :
    CandCombinerBase( ), 
    select_( select ), setup_( setup ) { }
  /// constructor from a selector and two charges
  CandCombiner( const S & select, int q1, int q2, const Setup & setup ) :
    CandCombinerBase( q1, q2 ), 
    select_( select ), setup_( setup ) { }
  /// constructor from a selector and three charges
  CandCombiner( const S & select, int q1, int q2, int q3, const Setup & setup ) :
    CandCombinerBase( q1, q2, q3 ), 
    select_( select ), setup_( setup ) { }
  /// constructor from a selector and four charges
  CandCombiner( const S & select, int q1, int q2, int q3, int q4, const Setup & setup ) :
    CandCombinerBase( q1, q2, q3, q4 ), 
    select_( select ), setup_( setup ) { }
  /// constructor from a selector, specifying optionally to check for charge
  CandCombiner( const S & select, const Setup & setup,
		 bool checkCharge, const std::vector <int> & dauCharge ) : 
    CandCombinerBase( checkCharge, dauCharge ), 
    select_( select ), setup_( setup ) { }
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
