#ifndef CandUtils_NamedCandCombiner_h
#define CandUtils_NamedCandCombiner_h
/** \class NamedCandCombiner
 *
 * \author Luca Lista, INFN
 *
 */
#include "CommonTools/CandUtils/interface/NamedCandCombinerBase.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "CommonTools/CandUtils/interface/CandSelector.h"
#include "CommonTools/UtilAlgos/interface/AnyPairSelector.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"
#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"
#include <string>

namespace combiner {
  namespace helpers {
    struct NormalClone {
      template<typename Ptr, typename CMP>
	static void addDaughter(CMP & cmp, const Ptr & c, std::string name) {
	cmp.addDaughter(*c, name);
      }
    };
    
    struct ShallowClone {
      template<typename CMP>
      static void addDaughter(CMP & cmp, const reco::CandidatePtr & c, std::string name) {
	if(c->numberOfDaughters()==0)
	  cmp.addDaughter(reco::ShallowClonePtrCandidate(c), name);
	else
	  cmp.addDaughter(*c, name);
      }
    };
  }
}

template<typename Selector, 
	 typename PairSelector = AnyPairSelector,
	 typename Cloner = combiner::helpers::NormalClone, 
	 typename Setup = AddFourMomenta>
class NamedCandCombiner : public NamedCandCombinerBase {
public:
  /// default constructor
  NamedCandCombiner(std::string name) :
    NamedCandCombinerBase(name), 
    select_(), selectPair_(), setup_(), name_(name) { }
  /// constructor from a selector and two charges
  NamedCandCombiner(std::string name, int q1, int q2) :
    NamedCandCombinerBase(name, q1, q2), 
    select_(), selectPair_(), setup_(), name_(name) { }
  /// constructor from a selector and three charges
  NamedCandCombiner(std::string name,  int q1, int q2, int q3 ) :
    NamedCandCombinerBase(name, q1, q2, q3), 
    select_(), selectPair_(), setup_(), name_(name) { }
  /// constructor from a selector and four charges
  NamedCandCombiner(std::string name, int q1, int q2, int q3, int q4) :
    NamedCandCombinerBase(name, q1, q2, q3, q4), 
    select_(), selectPair_(), setup_(), name_(name) { }
  /// default constructor
  NamedCandCombiner(std::string name, const Selector & select) :
    NamedCandCombinerBase(name), 
    select_(select), selectPair_(), setup_(), name_(name) { }
  /// constructor from a selector and two charges
  NamedCandCombiner(std::string name, const Selector & select, int q1, int q2 ) :
    NamedCandCombinerBase(name, q1, q2), 
    select_(select), selectPair_(), setup_(), name_(name) { }
  /// constructor from a selector and three charges
  NamedCandCombiner(std::string name,  const Selector & select, int q1, int q2, int q3 ) :
    NamedCandCombinerBase(name, q1, q2, q3), 
    select_(select), selectPair_(), setup_(), name_(name) { }
  /// constructor from a selector and four charges
  NamedCandCombiner(std::string name,  const Selector & select, int q1, int q2, int q3, int q4 ) :
    NamedCandCombinerBase(name, q1, q2, q3, q4), 
    select_(select), selectPair_(), setup_(), name_(name) { }
  /// constructor from selector
  NamedCandCombiner(std::string name, const Selector & select, const PairSelector & selectPair) :
    NamedCandCombinerBase(name), 
    select_(select), selectPair_(selectPair), setup_(), name_(name) { }
  /// constructor from a selector and two charges
  NamedCandCombiner(std::string name, const Selector & select, const PairSelector & selectPair, int q1, int q2) :
    NamedCandCombinerBase(name, q1, q2), 
    select_(select), selectPair_(selectPair), setup_(), name_(name) { }
  /// constructor from a selector and three charges
  NamedCandCombiner(std::string name, const Selector & select, const PairSelector & selectPair, int q1, int q2, int q3) :
    NamedCandCombinerBase(name, q1, q2, q3), 
    select_(select), selectPair_(selectPair), setup_(), name_(name) { }
  /// constructor from a selector and four charges
  NamedCandCombiner(std::string name, const Selector & select, const PairSelector & selectPair, int q1, int q2, int q3, int q4) :
   NamedCandCombinerBase(name, q1, q2, q3, q4), 
    select_(select), selectPair_(selectPair), setup_(), name_(name) { }
  NamedCandCombiner(std::string name, const Selector & select, const PairSelector & selectPair, const Setup & setup) :
    NamedCandCombinerBase(name), 
    select_(select), selectPair_(selectPair), setup_(setup), name_(name) { }
  /// constructor from a selector and two charges
  NamedCandCombiner(std::string name, const Selector & select, const PairSelector & selectPair, const Setup & setup, int q1, int q2) :
    NamedCandCombinerBase(name, q1, q2), 
    select_(select), selectPair_(selectPair), setup_(setup), name_(name) { }
  /// constructor from a selector and three charges
  NamedCandCombiner(std::string name, const Selector & select, const PairSelector & selectPair, const Setup & setup, int q1, int q2, int q3) :
    NamedCandCombinerBase(name, q1, q2, q3), 
    select_(select), selectPair_(selectPair), setup_(setup), name_(name) { }
  /// constructor from a selector and four charges
  NamedCandCombiner(std::string name, const Selector & select, const PairSelector & selectPair, const Setup & setup, int q1, int q2, int q3, int q4) :
    NamedCandCombinerBase(name, q1, q2, q3, q4), 
    select_(select), selectPair_(selectPair), setup_(setup), name_(name) { }
  /// constructor from a selector, specifying to check for charge
  NamedCandCombiner(std::string name, const Selector & select, const PairSelector & selectPair, const Setup & setup,const std::vector <int> & dauCharge) : 
    NamedCandCombinerBase(name, true, dauCharge), select_(select), selectPair_(selectPair), setup_(setup), name_(name) { }
  /// constructor from a selector, specifying to check for charge
  NamedCandCombiner( std::string name, const Selector & select, const PairSelector & selectPair, const std::vector <int> & dauCharge ) : 
    NamedCandCombinerBase(name, true, dauCharge), select_(select), selectPair_(selectPair), setup_(), name_(name) { }
  /// constructor from a selector, specifying to check for charge
  NamedCandCombiner(std::string name, const std::vector <int> & dauCharge) : 
    NamedCandCombinerBase(name, true, dauCharge), select_(), selectPair_(), setup_(), name_(name) { }
  /// constructor from a selector, specifying optionally to check for charge
  NamedCandCombiner(std::string name, const Selector & select, const PairSelector & selectPair, const Setup & setup,
	       bool checkCharge, const std::vector <int> & dauCharge) : 
    NamedCandCombinerBase(name, checkCharge, dauCharge), 
    select_(select), selectPair_(selectPair), setup_(setup), name_(name) { }
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
  virtual void setup(reco::NamedCompositeCandidate & c) const {
    setup_.set(c);
  }
  /// add candidate daughter
  virtual void addDaughter(reco::NamedCompositeCandidate & cmp, const reco::CandidatePtr & c, std::string n) const {
    Cloner::addDaughter(cmp, c, n);
  }
  /// candidate selector
  Selector select_; 
  /// candidate pair selector
  PairSelector selectPair_; 
  /// utility to setup composite candidate kinematics from daughters
  Setup setup_;
  /// name
  std::string name_;
};

#endif
