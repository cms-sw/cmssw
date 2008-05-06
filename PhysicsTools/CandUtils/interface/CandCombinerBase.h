#ifndef PhysicsTools_CandUtils_CandCombinerBase_h
#define PhysicsTools_CandUtils_CandCombinerBase_h
/** \class CandCombinerBase
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include <vector>
#include <utility>

namespace combiner {
  namespace helpers {
    template<typename OutputCollection>
      struct CandCompHelper {
	typedef typename OutputCollection::value_type composite_type;
	typedef composite_type value_type;
	static composite_type & make(value_type & v) { return v; }
      };  

    template<>
      struct CandCompHelper<reco::CandidateCollection> {
	typedef reco::CompositeCandidate composite_type;
	typedef reco::CompositeCandidate * value_type;
	static composite_type & make(value_type & v) { 
	  v = new composite_type; return *v; 
	}
      };
  }
}

template<typename OutputCollection = reco::CompositeCandidateCollection>
class CandCombinerBase {
public:
  typedef combiner::helpers::CandCompHelper<OutputCollection> OutputHelper;
  typedef typename OutputHelper::composite_type composite_type;
  typedef typename OutputHelper::value_type value_type;
  /// default construct
  CandCombinerBase();
  /// construct from two charge values
  CandCombinerBase(int, int);
  /// construct from three charge values
  CandCombinerBase(int, int, int);
  /// construct from four charge values
  CandCombinerBase(int, int, int, int);
  /// constructor from a selector, specifying optionally to check for charge
  CandCombinerBase(bool checkCharge, const std::vector <int> &);
  /// destructor
  virtual ~CandCombinerBase();
  /// return all selected candidate pairs
  std::auto_ptr<OutputCollection> 
  combine(const std::vector<reco::CandidateBaseRefProd> &) const;
  /// return all selected candidate pairs
  std::auto_ptr<OutputCollection> 
  combine(const reco::CandidateBaseRefProd &) const;
  /// return all selected candidate pairs
  std::auto_ptr<OutputCollection> 
  combine(const reco::CandidateBaseRefProd &, const reco::CandidateBaseRefProd &) const;
  /// return all selected candidate pairs
  std::auto_ptr<OutputCollection> 
  combine(const reco::CandidateBaseRefProd &, const reco::CandidateBaseRefProd &, const reco::CandidateBaseRefProd &) const;
  /// return all selected candidate pairs
  std::auto_ptr<OutputCollection> 
  combine(const reco::CandidateBaseRefProd &, const reco::CandidateBaseRefProd &, const reco::CandidateBaseRefProd &, 
	  const reco::CandidateBaseRefProd &) const;

private:
  /// verify that the two candidate don't overlap and check charge
  bool preselect(const reco::Candidate &, const reco::Candidate &) const;
  /// returns a composite candidate combined from two daughters
  void combine(composite_type & , const reco::CandidateBaseRef &, const reco::CandidateBaseRef &) const;
  /// temporary candidate stack
  typedef std::vector<std::pair<std::pair<reco::CandidateBaseRef, size_t>, 
    typename std::vector<reco::CandidateBaseRefProd>::const_iterator> > CandStack;
  typedef std::vector<int> ChargeStack;
  /// returns a composite candidate combined from two daughters
  void combine(size_t collectionIndex, CandStack &, ChargeStack &,
	       typename std::vector<reco::CandidateBaseRefProd>::const_iterator begin,
	       typename std::vector<reco::CandidateBaseRefProd>::const_iterator end,
	       std::auto_ptr<OutputCollection> & comps
	       ) const;
  /// select a candidate
  virtual bool select(const reco::Candidate &) const = 0;
  /// select a candidate pair
  virtual bool selectPair(const reco::Candidate & c1, const reco::Candidate & c2) const = 0;
  /// set kinematics to reconstructed composite
  virtual void setup(composite_type &) const = 0;
  /// add candidate daughter
  virtual void addDaughter(composite_type & cmp, const reco::CandidateBaseRef & c) const = 0;
  /// flag to specify the checking of electric charge
  bool checkCharge_;
  /// electric charges of the daughters
  std::vector<int> dauCharge_;
  /// utility to check candidate daughters overlap
  OverlapChecker overlap_;
};

template<typename OutputCollection>
CandCombinerBase<OutputCollection>::CandCombinerBase() :
  checkCharge_(false), dauCharge_(), overlap_() {
}

template<typename OutputCollection>
CandCombinerBase<OutputCollection>::CandCombinerBase(int q1, int q2) :
  checkCharge_(true), dauCharge_(2), overlap_() {
  dauCharge_[0] = q1;
  dauCharge_[1] = q2;
}

template<typename OutputCollection>
CandCombinerBase<OutputCollection>::CandCombinerBase(int q1, int q2, int q3) :
  checkCharge_(true), dauCharge_(3), overlap_() {
  dauCharge_[0] = q1;
  dauCharge_[1] = q2;
  dauCharge_[2] = q3;
}

template<typename OutputCollection>
CandCombinerBase<OutputCollection>::CandCombinerBase(int q1, int q2, int q3, int q4) :
  checkCharge_(true), dauCharge_(4), overlap_() {
  dauCharge_[0] = q1;
  dauCharge_[1] = q2;
  dauCharge_[2] = q3;
  dauCharge_[3] = q4;
}

template<typename OutputCollection>
CandCombinerBase<OutputCollection>::CandCombinerBase(bool checkCharge, const std::vector<int> & dauCharge) :
  checkCharge_(checkCharge), dauCharge_(dauCharge), overlap_() {
}

template<typename OutputCollection>
CandCombinerBase<OutputCollection>::~CandCombinerBase() {
}

template<typename OutputCollection>
bool CandCombinerBase<OutputCollection>::preselect(const reco::Candidate & c1, const reco::Candidate & c2) const {
  if (checkCharge_) {
    int dq1 = dauCharge_[0], dq2 = dauCharge_[1], q1 = c1.charge(), q2 = c2.charge();
    bool matchCharge = (q1 == dq1 && q2 == dq2) || (q1 == -dq1 && q2 == -dq2); 
    if (!matchCharge) return false; 
  }
  if (overlap_(c1, c2)) return false;
  return selectPair(c1, c2);
}

template<typename OutputCollection>
void CandCombinerBase<OutputCollection>::combine(composite_type & cmp, const reco::CandidateBaseRef & c1, const reco::CandidateBaseRef & c2) const {
  addDaughter(cmp, c1);
  addDaughter(cmp, c2);
  setup(cmp);
}

template<typename OutputCollection>
std::auto_ptr<OutputCollection> 
CandCombinerBase<OutputCollection>::combine(const std::vector<reco::CandidateBaseRefProd> & src) const {
  size_t srcSize = src.size();
  if (checkCharge_ && dauCharge_.size() != srcSize)
    throw edm::Exception(edm::errors::Configuration) 
      << "CandCombiner: trying to combine " << srcSize << " collections"
      << " but configured to check against " << dauCharge_.size() << " charges.";
  
  std::auto_ptr<OutputCollection> comps(new OutputCollection);
  if(srcSize == 2) {
    reco::CandidateBaseRefProd src1 = src[0], src2 = src[1];
    if(src1 == src2) {
      const reco::CandidateView & cands = * src1;
      const int n = cands.size();
      for(int i1 = 0; i1 < n; ++i1) {
	const reco::Candidate & c1 = cands[i1];
	reco::CandidateBaseRef cr1(src1, i1);
	for(int i2 = i1 + 1; i2 < n; ++i2) {
	  const reco::Candidate & c2 = cands[i2];
	  if (preselect(c1, c2)) {
	    reco::CandidateBaseRef cr2(src2, i2);
	    value_type c; 
	    composite_type & r = OutputHelper::make(c);
	    combine(r, cr1, cr2);
	    if(select(r))
	      comps->push_back(c);
	  }
	}
      }
    } else {
      const reco::CandidateView & cands1 = * src1, & cands2 = * src2;
      const int n1 = cands1.size(), n2 = cands2.size();
      for(int i1 = 0; i1 < n1; ++i1) {
	const reco::Candidate & c1 = cands1[i1];
	reco::CandidateBaseRef cr1(src1, i1);
	for(int i2 = 0; i2 < n2; ++i2) {
	  const reco::Candidate & c2 = cands2[i2];
	  if(preselect(c1, c2)) {
	    reco::CandidateBaseRef cr2(src2, i2);
	    value_type c;
	    composite_type & r = OutputHelper::make(c);
	    combine(r, cr1, cr2);
	    if(select(r))
	      comps->push_back(c);
	  }
	}
      }
    }
  } else {
    CandStack stack;
    ChargeStack qStack;
    combine(0, stack, qStack, src.begin(), src.end(), comps);
  }

  return comps;
}

template<typename OutputCollection>
std::auto_ptr<OutputCollection> 
CandCombinerBase<OutputCollection>::combine(const reco::CandidateBaseRefProd & src) const {
  if(checkCharge_ && dauCharge_.size() != 2)
    throw edm::Exception(edm::errors::Configuration) 
      << "CandCombiner: trying to combine 2 collections"
      << " but configured to check against " << dauCharge_.size() << " charges.";

  std::auto_ptr<OutputCollection> comps(new OutputCollection);
  const reco::CandidateView & cands = * src; 
  const int n = cands.size();
  for(int i1 = 0; i1 < n; ++i1) {
    const reco::Candidate & c1 = cands[i1];
    reco::CandidateBaseRef cr1(src, i1);
    for(int i2 = i1 + 1; i2 < n; ++i2) {
      const reco::Candidate & c2 = cands[i2];
      if(preselect(c1, c2)) {
	reco::CandidateBaseRef cr2(src, i2);
	value_type c;
	composite_type & r = OutputHelper::make(c);
	combine(r, cr1, cr2);
	if(select(r))
	  comps->push_back(c);
      }
    } 
  }

  return comps;
}

template<typename OutputCollection>
std::auto_ptr<OutputCollection> 
CandCombinerBase<OutputCollection>::combine(const reco::CandidateBaseRefProd & src1, const reco::CandidateBaseRefProd & src2) const {
  std::vector<reco::CandidateBaseRefProd> src;
  src.push_back(src1);
  src.push_back(src2);
  return combine(src);
}

template<typename OutputCollection>
std::auto_ptr<OutputCollection> 
CandCombinerBase<OutputCollection>::combine(const reco::CandidateBaseRefProd & src1, const reco::CandidateBaseRefProd & src2, const reco::CandidateBaseRefProd & src3) const {
  std::vector<reco::CandidateBaseRefProd> src;
  src.push_back(src1);
  src.push_back(src2);
  src.push_back(src3);
  return combine(src);
}

template<typename OutputCollection>
std::auto_ptr<OutputCollection> 
CandCombinerBase<OutputCollection>::combine(const reco::CandidateBaseRefProd & src1, const reco::CandidateBaseRefProd & src2, 
					    const reco::CandidateBaseRefProd & src3, const reco::CandidateBaseRefProd & src4) const {
  std::vector<reco::CandidateBaseRefProd> src;
  src.push_back(src1);
  src.push_back(src2);
  src.push_back(src3);
  src.push_back(src4);
  return combine(src);
}

template<typename OutputCollection>
void CandCombinerBase<OutputCollection>::combine(size_t collectionIndex, CandStack & stack, ChargeStack & qStack,
						 typename std::vector<reco::CandidateBaseRefProd>::const_iterator collBegin,
						 typename std::vector<reco::CandidateBaseRefProd>::const_iterator collEnd,
						 std::auto_ptr<OutputCollection> & comps) const {
  if(collBegin == collEnd) {
    static const int undetermined = 0, sameDecay = 1, conjDecay = -1, wrongDecay = 2;
    int decayType = undetermined;
    if(checkCharge_) {
      assert(qStack.size() == stack.size());
      for(size_t i = 0; i < qStack.size(); ++i) {
	int q = qStack[i], dq = dauCharge_[i];
	if(decayType == undetermined) {
	  if(q != 0 && dq != 0) {
	    if(q == dq) decayType = sameDecay;
	    else if(q == -dq) decayType = conjDecay;
	    else decayType = wrongDecay;
	  }
	} else if((decayType == sameDecay && q != dq) ||
		  (decayType == conjDecay && q != -dq)) {
	  decayType = wrongDecay;
	}
	if(decayType == wrongDecay) break;
      }
    }
    if(decayType != wrongDecay) { 
      value_type c;
      composite_type & r = OutputHelper::make(c);
      for(typename CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i) {
	addDaughter(r, i->first.first);
      }
      setup(r);
      if(select(r))
	comps->push_back(c);
    }
  } else {
    const reco::CandidateBaseRefProd & srcRef = * collBegin;
    const reco::CandidateView & src = * srcRef;
    size_t candBegin = 0, candEnd = src.size();
    for(typename CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i) 
      if(srcRef == * i->second) 
	candBegin = i->first.second + 1;
    for(size_t candIndex = candBegin; candIndex != candEnd; ++ candIndex) {
      reco::CandidateBaseRef candRef(srcRef, candIndex);
      bool noOverlap = true;
      const reco::Candidate & cand = * candRef;
      for(typename CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i) 
	if(overlap_(cand, *(i->first.first))) { 
	  noOverlap = false; 
	  break; 
	}
      if(noOverlap) {
	stack.push_back(std::make_pair(std::make_pair(candRef, candIndex), collBegin));
	if(checkCharge_) qStack.push_back(cand.charge()); 
	combine(collectionIndex + 1, stack, qStack, collBegin + 1, collEnd, comps);
	stack.pop_back();
	qStack.pop_back();
      }
    }
  }
}

#endif
