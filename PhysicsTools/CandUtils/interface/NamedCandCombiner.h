#ifndef CandUtils_NamedCandCombiner_h
#define CandUtils_NamedCandCombiner_h
/** \class NamedCandCombiner
 *
 * \author Luca Lista, INFN
 *
 */
#include "PhysicsTools/CandUtils/interface/NamedCandCombinerBase.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "PhysicsTools/UtilAlgos/interface/AnyPairSelector.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"

#include <string>

namespace combiner {
  namespace helpers {
    struct NormalClone {
      template<typename Ref, typename CMP>
	static void addDaughter(CMP & cmp, const Ref & c, std::string name) {
	cmp.addDaughter(*c, name);
      }
    };
    
    struct ShallowClone {
      template<typename CMP>
      static void addDaughter(CMP & cmp, const reco::CandidateRef & c, std::string name) {
	if(c->numberOfDaughters()==0)
	  cmp.addDaughter(reco::ShallowCloneCandidate(reco::CandidateBaseRef(c)), name);
	else
	  cmp.addDaughter(*c, name);
      }
      template<typename CMP>
      static void addDaughter(CMP & cmp, const reco::CandidateBaseRef & c, std::string name) {
	if(c->numberOfDaughters()==0)
	  cmp.addDaughter(reco::ShallowCloneCandidate(c), name);
	else
	  cmp.addDaughter(*c, name);
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
class NamedCandCombiner : public NamedCandCombinerBase<InputCollection, OutputCollection> {
public:
  typedef NamedCandCombinerBase<InputCollection, OutputCollection> base;
  /// default constructor
  NamedCandCombiner(std::string name = "") :
  base(name), 
    select_(), selectPair_(), setup_() { }
  /// constructor from a selector and two charges
  NamedCandCombiner(std::string name, int q1, int q2) :
    base(name, q1, q2), 
    select_(), selectPair_(), setup_() { }
  /// constructor from a selector and three charges
  NamedCandCombiner(std::string name, int q1, int q2, int q3 ) :
    base(name, q1, q2, q3), 
    select_(), selectPair_(), setup_() { }
  /// constructor from a selector and four charges
  NamedCandCombiner(std::string name,int q1, int q2, int q3, int q4) :
    base(name, q1, q2, q3, q4), 
    select_(), selectPair_(), setup_() { }
  /// default constructor
  NamedCandCombiner(std::string name,const Selector & select) :
    base(name ), 
    select_(select), selectPair_(), setup_() { }
  /// constructor from a selector and two charges
  NamedCandCombiner(std::string name, const Selector & select, int q1, int q2 ) :
    base(name, q1, q2), 
    select_(select), selectPair_(), setup_() { }
  /// constructor from a selector and three charges
  NamedCandCombiner(std::string name, const Selector & select, int q1, int q2, int q3 ) :
    base(name, q1, q2, q3), 
    select_(select), selectPair_(), setup_() { }
  /// constructor from a selector and four charges
  NamedCandCombiner(std::string name, const Selector & select, int q1, int q2, int q3, int q4 ) :
    base(name, q1, q2, q3, q4), 
    select_(select), selectPair_(), setup_() { }
  /// constructor from selector
  NamedCandCombiner(std::string name,const Selector & select, const PairSelector & selectPair) :
    base(name ), 
    select_(select), selectPair_(selectPair), setup_() { }
  /// constructor from a selector and two charges
  NamedCandCombiner(std::string name,const Selector & select, const PairSelector & selectPair, int q1, int q2) :
    base(name, q1, q2), 
    select_(select), selectPair_(selectPair), setup_() { }
  /// constructor from a selector and three charges
  NamedCandCombiner(std::string name,const Selector & select, const PairSelector & selectPair, int q1, int q2, int q3) :
    base(name, q1, q2, q3), 
    select_(select), selectPair_(selectPair), setup_() { }
  /// constructor from a selector and four charges
  NamedCandCombiner(std::string name,const Selector & select, const PairSelector & selectPair, int q1, int q2, int q3, int q4) :
    base(name, q1, q2, q3, q4), 
    select_(select), selectPair_(selectPair), setup_() { }
  NamedCandCombiner(std::string name,const Selector & select, const PairSelector & selectPair, const Setup & setup) :
    base(name ), 
    select_(select), selectPair_(selectPair), setup_(setup) { }
  /// constructor from a selector and two charges
  NamedCandCombiner(std::string name,const Selector & select, const PairSelector & selectPair, const Setup & setup, int q1, int q2) :
    base(name, q1, q2), 
    select_(select), selectPair_(selectPair), setup_(setup) { }
  /// constructor from a selector and three charges
  NamedCandCombiner(std::string name,const Selector & select, const PairSelector & selectPair, const Setup & setup, int q1, int q2, int q3) :
    base(name, q1, q2, q3), 
    select_(select), selectPair_(selectPair), setup_(setup) { }
  /// constructor from a selector and four charges
  NamedCandCombiner(std::string name,const Selector & select, const PairSelector & selectPair, const Setup & setup, int q1, int q2, int q3, int q4) :
    base(name, q1, q2, q3, q4), 
    select_(select), selectPair_(selectPair), setup_(setup) { }
  /// constructor from a selector, specifying to check for charge
  NamedCandCombiner(std::string name,const Selector & select, const PairSelector & selectPair, const Setup & setup,const std::vector <int> & dauCharge) : 
    base(name, true, dauCharge), select_(select), selectPair_(selectPair), setup_(setup) { }
  /// constructor from a selector, specifying to check for charge
  NamedCandCombiner(std::string name, const Selector & select, const PairSelector & selectPair, const std::vector <int> & dauCharge ) : 
    base(name, true, dauCharge), select_(select), selectPair_(selectPair), setup_() { }
  /// constructor from a selector, specifying to check for charge
  NamedCandCombiner(std::string name,const std::vector <int> & dauCharge) : 
    base(name, true, dauCharge), select_(), selectPair_(), setup_() { }
  /// constructor from a selector, specifying optionally to check for charge
  NamedCandCombiner(std::string name,const Selector & select, const PairSelector & selectPair, const Setup & setup,
	       bool checkCharge, const std::vector <int> & dauCharge) : 
    base(name, checkCharge, dauCharge), 
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
    c.setName(name_);
    setup_.set(c);
  }
  /// add candidate daughter
  virtual void addDaughter(composite_type & cmp, const Ref & c, std::string name) const {
    Cloner::addDaughter(cmp, c, name);
  }
  /// candidate selector
  Selector select_; 
  /// candidate pair selector
  PairSelector selectPair_; 
  /// utility to setup composite candidate kinematics from daughters
  Setup setup_;
  /// Name
  std::string name_;
};

#endif
