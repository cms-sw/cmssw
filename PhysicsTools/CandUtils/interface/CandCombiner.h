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
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"

namespace combiner {
  namespace helpers {
    struct NormalClone {
      typedef reco::CandidateBaseRef CandPtr;
      template<typename Ref, typename CMP>
      static void addDaughter(CMP & cmp, const Ref & c) {
	cmp.addDaughter(*c);
      }
    };
     
    struct ShallowClone {
      typedef reco::CandidateBaseRef CandPtr;
      template<typename CMP>
      static void addDaughter(CMP & cmp, const reco::CandidateBaseRef & c) {
	if(c->numberOfDaughters()==0)
	  cmp.addDaughter(reco::ShallowCloneCandidate(c));
	else
	  cmp.addDaughter(*c);
      }
    };
    struct ShallowClonePtr {
      typedef reco::CandidatePtr CandPtr;
      template<typename CMP>
      static void addDaughter(CMP & cmp, const reco::CandidatePtr & c) {
	if(c->numberOfDaughters()==0)
	  cmp.addDaughter(reco::ShallowClonePtrCandidate(c));
	else
	  cmp.addDaughter(*c);
      }
    };
 }
}

template<typename Selector, 
	 typename PairSelector = AnyPairSelector,
	 typename Cloner = combiner::helpers::NormalClone, 
	 typename OutputCollection = reco::CompositeCandidateCollection,
	 typename Setup = AddFourMomenta>
class CandCombiner : public CandCombinerBase<OutputCollection, typename Cloner::CandPtr> {
public:
  typedef typename Cloner::CandPtr CandPtr;
  typedef CandCombinerBase<OutputCollection, CandPtr> base;
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
  /// select a candidate
  virtual bool select(const reco::Candidate & c) const {
    return select_(c);
  } 
  /// select a candidate
  virtual bool selectPair(const reco::Candidate & c1, const reco::Candidate & c2) const {
    return selectPair_(c1, c2);
  } 
  /// set kinematics to reconstructed composite
  virtual void setup(reco::CompositeCandidate & c) const {
    setup_.set(c);
  }
  /// add candidate daughter
  virtual void addDaughter(reco::CompositeCandidate & cmp, const CandPtr & c) const {
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
