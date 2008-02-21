#ifndef CandUtils_CandCombiner_h
#define CandUtils_CandCombiner_h
/** \class CandCombiner
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/CandUtils/interface/CandCombinerBase.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/UtilAlgos/interface/AnyPairSelector.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"

namespace combiner {
  namespace helpers {
    struct NormalClone {
      template<typename Ref, typename CMP>
      static void addDaughter(CMP & cmp, const Ref & c) {
	cmp.addDaughter(*c);
      }
    };
    
    struct ShallowClone {
      template<typename CMP>
      static void addDaughter(CMP & cmp, const reco::CandidateRef & c) {
	if(c->numberOfDaughters()==0)
	  cmp.addDaughter(reco::ShallowCloneCandidate(reco::CandidateBaseRef(c)));
	else
	  cmp.addDaughter(*c);
      }
      template<typename CMP>
      static void addDaughter(CMP & cmp, const reco::CandidateBaseRef & c) {
	if(c->numberOfDaughters()==0)
	  cmp.addDaughter(reco::ShallowCloneCandidate(c));
	else
	  cmp.addDaughter(*c);
      }
    };
  }
}

template<typename InputCollection,
         typename Selector, 
	 typename OutputCollection = typename combiner::helpers::CandRefHelper<InputCollection>::OutputCollection,
	 typename PairSelector = AnyPairSelector,
	 typename Cloner = combiner::helpers::NormalClone, 
	 typename Setup = AddFourMomenta>
class CandCombiner : public CandCombinerBase<InputCollection, OutputCollection> {
public:
  typedef CandCombinerBase<InputCollection, OutputCollection> base;
  /// default constructor
  CandCombiner() :
  base(), 
    select_(), selectPair_(), setup_() { }
  /// constructor from a selector and two charges
  CandCombiner(int q1, int q2) :
    base(q1, q2), 
    select_(), selectPair_(), setup_() { }
  /// constructor from a selector and three charges
  CandCombiner( int q1, int q2, int q3 ) :
    base(q1, q2, q3), 
    select_(), selectPair_(), setup_() { }
  /// constructor from a selector and four charges
  CandCombiner(int q1, int q2, int q3, int q4) :
    base(q1, q2, q3, q4), 
    select_(), selectPair_(), setup_() { }
  /// default constructor
  CandCombiner(const Selector & select) :
    base( ), 
    select_(select), selectPair_(), setup_() { }
  /// constructor from a selector and two charges
  CandCombiner( const Selector & select, int q1, int q2 ) :
    base(q1, q2), 
    select_(select), selectPair_(), setup_() { }
  /// constructor from a selector and three charges
  CandCombiner( const Selector & select, int q1, int q2, int q3 ) :
    base(q1, q2, q3), 
    select_(select), selectPair_(), setup_() { }
  /// constructor from a selector and four charges
  CandCombiner( const Selector & select, int q1, int q2, int q3, int q4 ) :
    base(q1, q2, q3, q4), 
    select_(select), selectPair_(), setup_() { }
  /// constructor from selector
  CandCombiner(const Selector & select, const PairSelector & selectPair) :
    base( ), 
    select_(select), selectPair_(selectPair), setup_() { }
  /// constructor from a selector and two charges
  CandCombiner(const Selector & select, const PairSelector & selectPair, int q1, int q2) :
    base(q1, q2), 
    select_(select), selectPair_(selectPair), setup_() { }
  /// constructor from a selector and three charges
  CandCombiner(const Selector & select, const PairSelector & selectPair, int q1, int q2, int q3) :
    base(q1, q2, q3), 
    select_(select), selectPair_(selectPair), setup_() { }
  /// constructor from a selector and four charges
  CandCombiner(const Selector & select, const PairSelector & selectPair, int q1, int q2, int q3, int q4) :
    base(q1, q2, q3, q4), 
    select_(select), selectPair_(selectPair), setup_() { }
  CandCombiner(const Selector & select, const PairSelector & selectPair, const Setup & setup) :
    base(), 
    select_(select), selectPair_(selectPair), setup_(setup) { }
  /// constructor from a selector and two charges
  CandCombiner(const Selector & select, const PairSelector & selectPair, const Setup & setup, int q1, int q2) :
    base(q1, q2), 
    select_(select), selectPair_(selectPair), setup_(setup) { }
  /// constructor from a selector and three charges
  CandCombiner(const Selector & select, const PairSelector & selectPair, const Setup & setup, int q1, int q2, int q3) :
    base(q1, q2, q3), 
    select_(select), selectPair_(selectPair), setup_(setup) { }
  /// constructor from a selector and four charges
  CandCombiner(const Selector & select, const PairSelector & selectPair, const Setup & setup, int q1, int q2, int q3, int q4) :
    base(q1, q2, q3, q4), 
    select_(select), selectPair_(selectPair), setup_(setup) { }
  /// constructor from a selector, specifying to check for charge
  CandCombiner(const Selector & select, const PairSelector & selectPair, const Setup & setup,const std::vector <int> & dauCharge) : 
    base(true, dauCharge), select_(select), selectPair_(selectPair), setup_(setup) { }
  /// constructor from a selector, specifying to check for charge
  CandCombiner( const Selector & select, const PairSelector & selectPair, const std::vector <int> & dauCharge ) : 
    base(true, dauCharge), select_(select), selectPair_(selectPair), setup_() { }
  /// constructor from a selector, specifying to check for charge
  CandCombiner(const std::vector <int> & dauCharge) : 
    base(true, dauCharge), select_(), selectPair_(), setup_() { }
  /// constructor from a selector, specifying optionally to check for charge
  CandCombiner(const Selector & select, const PairSelector & selectPair, const Setup & setup,
	       bool checkCharge, const std::vector <int> & dauCharge) : 
    base(checkCharge, dauCharge), 
    select_(select), selectPair_(selectPair), setup_(setup) { }
  /// return reference to setup object to allow its initialization
  Setup & setup() { return setup_; }

private:
  typedef typename base::Ref Ref;
  typedef typename base::composite_type composite_type;

  /// select a candidate
  virtual bool select(const reco::Candidate & c) const {
    return select_(c);
  } 
  /// select a candidate
  virtual bool selectPair(const reco::Candidate & c1, const reco::Candidate & c2) const {
    return selectPair_(c1, c2);
  } 
  /// set kinematics to reconstructed composite
  virtual void setup(composite_type & c) const {
    setup_.set(c);
  }
  /// add candidate daughter
  virtual void addDaughter(composite_type & cmp, const Ref & c) const {
    Cloner::addDaughter(cmp, c);
  }
  /// candidate selector
  Selector select_; 
  /// candidate pair selector
  PairSelector selectPair_; 
  /// utility to setup composite candidate kinematics from daughters
  Setup setup_;
};

#endif
