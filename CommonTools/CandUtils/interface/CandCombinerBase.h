#ifndef CommonTools_CandUtils_CandCombinerBase_h
#define CommonTools_CandUtils_CandCombinerBase_h
/** \class CandCombinerBase
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include <vector>
#include <string>

template<typename OutputCollection, typename CandPtr>
class CandCombinerBase {
public:
  typedef std::vector<std::string> vstring;
  /// default construct
  explicit CandCombinerBase(const std::string  = "");
  /// construct from two charge values
  CandCombinerBase(int, int, const std::string  = "");
  /// construct from three charge values
  CandCombinerBase(int, int, int, const std::string  = "");
  /// construct from four charge values
  CandCombinerBase(int, int, int, int, const std::string  = "");
  /// constructor from a selector, specifying optionally to check for charge
  CandCombinerBase(bool checkCharge, const std::vector <int> &, const std::string  = "");
  /// destructor
  virtual ~CandCombinerBase();
  /// return all selected candidate pairs
  std::auto_ptr<OutputCollection> 
  combine(const std::vector<edm::Handle<reco::CandidateView> > &, const vstring = vstring()) const;
  /// return all selected candidate pairs
  std::auto_ptr<OutputCollection> 
  combine(const edm::Handle<reco::CandidateView> &, const vstring = vstring()) const;
  /// return all selected candidate pairs
  std::auto_ptr<OutputCollection> 
  combine(const edm::Handle<reco::CandidateView> &, 
	  const edm::Handle<reco::CandidateView> &, const vstring = vstring()) const;
  /// return all selected candidate pairs
  std::auto_ptr<OutputCollection> 
  combine(const edm::Handle<reco::CandidateView> &, 
	  const edm::Handle<reco::CandidateView> &, 
	  const edm::Handle<reco::CandidateView> &, const vstring = vstring()) const;
  /// return all selected candidate pairs
  std::auto_ptr<OutputCollection> 
  combine(const edm::Handle<reco::CandidateView> &, 
	  const edm::Handle<reco::CandidateView> &, 
	  const edm::Handle<reco::CandidateView> &, 
	  const edm::Handle<reco::CandidateView> &, const vstring = vstring()) const;

private:
  /// verify that the two candidate don't overlap and check charge
  bool preselect(const reco::Candidate &, const reco::Candidate &) const;
  /// returns a composite candidate combined from two daughters
  void combine(typename OutputCollection::value_type &, 
	       const CandPtr &, 
	       const CandPtr &, const std::string = "", const std::string = "") const;
  /// temporary candidate stack
  typedef std::vector<std::pair<std::pair<CandPtr, size_t>, 
				std::vector<edm::Handle<reco::CandidateView> >::const_iterator> > CandStack;
  typedef std::vector<int> ChargeStack;
  /// returns a composite candidate combined from two daughters
  void combine(size_t collectionIndex, CandStack &, ChargeStack &,
	       std::vector<edm::Handle<reco::CandidateView> >::const_iterator begin,
	       std::vector<edm::Handle<reco::CandidateView> >::const_iterator end,
	       std::auto_ptr<OutputCollection> & comps,
	       const vstring name = vstring()) const;
  /// select a candidate
  virtual bool select(const reco::Candidate &) const = 0;
  /// select a candidate pair
  virtual bool selectPair(const reco::Candidate & c1, const reco::Candidate & c2) const = 0;
  /// set kinematics to reconstructed composite
  virtual void setup(typename OutputCollection::value_type &) const = 0;
  /// add candidate daughter
  virtual void addDaughter(typename OutputCollection::value_type & cmp, const CandPtr & c, const std::string = "") const = 0;
  /// flag to specify the checking of electric charge
  bool checkCharge_;
  /// electric charges of the daughters
  std::vector<int> dauCharge_;
  /// utility to check candidate daughters overlap
  OverlapChecker overlap_;
  /// composite name (if applies)
  std::string name_;
};

template<typename OutputCollection, typename CandPtr>
CandCombinerBase<OutputCollection, CandPtr>::CandCombinerBase(const std::string name) :
  checkCharge_(false), dauCharge_(), overlap_(), name_(name) {
}

template<typename OutputCollection, typename CandPtr>
CandCombinerBase<OutputCollection, CandPtr>::CandCombinerBase(int q1, int q2, const std::string name) :
  checkCharge_(true), dauCharge_(2), overlap_(), name_(name) {
  dauCharge_[0] = q1;
  dauCharge_[1] = q2;
}

template<typename OutputCollection, typename CandPtr>
CandCombinerBase<OutputCollection, CandPtr>::CandCombinerBase(int q1, int q2, int q3, const std::string name) :
  checkCharge_(true), dauCharge_(3), overlap_(), name_(name) {
  dauCharge_[0] = q1;
  dauCharge_[1] = q2;
  dauCharge_[2] = q3;
}

template<typename OutputCollection, typename CandPtr>
CandCombinerBase<OutputCollection, CandPtr>::CandCombinerBase(int q1, int q2, int q3, int q4, const std::string name) :
  checkCharge_(true), dauCharge_(4), overlap_(), name_(name) {
  dauCharge_[0] = q1;
  dauCharge_[1] = q2;
  dauCharge_[2] = q3;
  dauCharge_[3] = q4;
}

template<typename OutputCollection, typename CandPtr>
CandCombinerBase<OutputCollection, CandPtr>::CandCombinerBase(bool checkCharge, const std::vector<int> & dauCharge, const std::string name) :
  checkCharge_(checkCharge), dauCharge_(dauCharge), overlap_(), name_(name) {
}

template<typename OutputCollection, typename CandPtr>
CandCombinerBase<OutputCollection, CandPtr>::~CandCombinerBase() {
}

template<typename OutputCollection, typename CandPtr>
bool CandCombinerBase<OutputCollection, CandPtr>::preselect(const reco::Candidate & c1, const reco::Candidate & c2) const {
  if (checkCharge_) {
    int dq1 = dauCharge_[0], dq2 = dauCharge_[1], q1 = c1.charge(), q2 = c2.charge();
    bool matchCharge = (q1 == dq1 && q2 == dq2) || (q1 == -dq1 && q2 == -dq2); 
    if (!matchCharge) return false; 
  }
  if (overlap_(c1, c2)) return false;
  return selectPair(c1, c2);
}

template<typename OutputCollection, typename CandPtr>
void CandCombinerBase<OutputCollection, CandPtr>::combine(typename OutputCollection::value_type & cmp, 
							  const CandPtr & c1, const CandPtr & c2,
							  const std::string name1, const std::string name2) const {
  addDaughter(cmp, c1, name1);
  addDaughter(cmp, c2, name2);
  setup(cmp);
}

template<typename OutputCollection, typename CandPtr>
std::auto_ptr<OutputCollection> 
CandCombinerBase<OutputCollection, CandPtr>::combine(const std::vector<edm::Handle<reco::CandidateView> > & src,
						     const vstring names) const {
  size_t srcSize = src.size();
  if (checkCharge_ && dauCharge_.size() != srcSize)
    throw edm::Exception(edm::errors::Configuration) 
      << "CandCombiner: trying to combine " << srcSize << " collections"
      << " but configured to check against " << dauCharge_.size() << " charges.";
  std::auto_ptr<OutputCollection> comps(new OutputCollection);
  size_t namesSize = names.size();
  if(srcSize == 2) {
    std::string name1="", name2="";
    if(namesSize > 0) {
      if(namesSize != 2)
	throw edm::Exception(edm::errors::Configuration)
	  << "CandCombiner: should specify exactly two "
	  << " names in configuration (" << namesSize << " specified).\n";
      name1 = names[0];
      name2 = names[1];
    }
    edm::Handle<reco::CandidateView> src1 = src[0], src2 = src[1];
    if(src1.id() == src2.id()) {
      const reco::CandidateView & cands = * src1;
      const size_t n = cands.size();
      for(size_t i1 = 0; i1 < n; ++i1) {
	const reco::Candidate & c1 = cands[i1];
	CandPtr cr1(src1, i1);
	for(size_t i2 = i1 + 1; i2 < n; ++i2) {
	  const reco::Candidate & c2 = cands[i2];
	  if(preselect(c1, c2)) {
	    CandPtr cr2(src2, i2);
	    typename OutputCollection::value_type c; 
	    combine(c, cr1, cr2, name1, name2);
	    if(select(c))
	      comps->push_back(c);
	  }
	}
      }
    } else {
      const reco::CandidateView & cands1 = * src1, & cands2 = * src2;
      const size_t n1 = cands1.size(), n2 = cands2.size();
      for(size_t i1 = 0; i1 < n1; ++i1) {
	const reco::Candidate & c1 = cands1[i1];
	CandPtr cr1(src1, i1);
	for(size_t i2 = 0; i2 < n2; ++i2) {
	  const reco::Candidate & c2 = cands2[i2];
	  if(preselect(c1, c2)) {
	    CandPtr cr2(src2, i2);
	    typename OutputCollection::value_type c;
	    combine(c, cr1, cr2, name1, name2);
	    if(select(c))
	      comps->push_back(c);
	  }
	}
      }
    }
  } else {
    CandStack stack;
    ChargeStack qStack;
    combine(0, stack, qStack, src.begin(), src.end(), comps, names);
  }

  return comps;
}

template<typename OutputCollection, typename CandPtr>
std::auto_ptr<OutputCollection> 
CandCombinerBase<OutputCollection, CandPtr>::combine(const edm::Handle<reco::CandidateView> & src,
						     const vstring names) const {
  if(checkCharge_ && dauCharge_.size() != 2)
    throw edm::Exception(edm::errors::Configuration) 
      << "CandCombiner: trying to combine 2 collections"
      << " but configured to check against " << dauCharge_.size() << " charges.";

  std::auto_ptr<OutputCollection> comps(new OutputCollection);
  size_t namesSize = names.size();
  std::string name1, name2;
  if(namesSize > 0) {
    if(namesSize != 2)
      throw edm::Exception(edm::errors::Configuration)
	<< "CandCombiner: should specify exactly two "
	<< " names in configuration (" << namesSize << " specified).\n";
    name1 = names[0];
    name2 = names[1];
  }
  const reco::CandidateView & cands = * src; 
  const size_t n = cands.size();
  for(size_t i1 = 0; i1 < n; ++i1) {
    const reco::Candidate & c1 = cands[i1];
    CandPtr cr1(src, i1);
    for(size_t i2 = i1 + 1; i2 < n; ++i2) {
      const reco::Candidate & c2 = cands[i2];
      if(preselect(c1, c2)) {
	CandPtr cr2(src, i2);
	typename OutputCollection::value_type c;
	combine(c, cr1, cr2, name1, name2);
	if(select(c))
	  comps->push_back(c);
      }
    } 
  }

  return comps;
}

template<typename OutputCollection, typename CandPtr>
std::auto_ptr<OutputCollection> 
CandCombinerBase<OutputCollection, CandPtr>::combine(const edm::Handle<reco::CandidateView> & src1, 
						     const edm::Handle<reco::CandidateView> & src2,
						     const vstring names) const {
  std::vector<edm::Handle<reco::CandidateView> > src;
  src.push_back(src1);
  src.push_back(src2);
  return combine(src, names);
}

template<typename OutputCollection, typename CandPtr>
std::auto_ptr<OutputCollection> 
CandCombinerBase<OutputCollection, CandPtr>::combine(const edm::Handle<reco::CandidateView> & src1, 
						     const edm::Handle<reco::CandidateView> & src2, 
						     const edm::Handle<reco::CandidateView> & src3,
						     const vstring names) const {
  std::vector<edm::Handle<reco::CandidateView> > src;
  src.push_back(src1);
  src.push_back(src2);
  src.push_back(src3);
  return combine(src, names);
}

template<typename OutputCollection, typename CandPtr>
std::auto_ptr<OutputCollection> 
CandCombinerBase<OutputCollection, CandPtr>::combine(const edm::Handle<reco::CandidateView> & src1, 
						     const edm::Handle<reco::CandidateView> & src2, 
						     const edm::Handle<reco::CandidateView> & src3, 
						     const edm::Handle<reco::CandidateView> & src4,
						     const vstring names) const {
  std::vector<edm::Handle<reco::CandidateView> > src;
  src.push_back(src1);
  src.push_back(src2);
  src.push_back(src3);
  src.push_back(src4);
  return combine(src, names);
}

template<typename OutputCollection, typename CandPtr>
void CandCombinerBase<OutputCollection, CandPtr>::combine(size_t collectionIndex, CandStack & stack, ChargeStack & qStack,
							  std::vector<edm::Handle<reco::CandidateView> >::const_iterator collBegin,
							  std::vector<edm::Handle<reco::CandidateView> >::const_iterator collEnd,
							  std::auto_ptr<OutputCollection> & comps,
							  const vstring names) const {
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
      typename OutputCollection::value_type c;
      size_t nameIndex = 0;
      for(typename CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i, ++ nameIndex) {
	if ( names.size() > 0 )
	  addDaughter(c, i->first.first, names[nameIndex]);
	else
	  addDaughter(c, i->first.first);	  
      }
      setup(c);
      if(select(c))
	comps->push_back(c);
    }
  } else {
    const edm::Handle<reco::CandidateView> & srcRef = * collBegin;
    const reco::CandidateView & src = * srcRef;
    size_t candBegin = 0, candEnd = src.size();
    for(typename CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i) 
      if(srcRef.id() == i->second->id()) 
	candBegin = i->first.second + 1;
    for(size_t candIndex = candBegin; candIndex != candEnd; ++ candIndex) {
      CandPtr candRef(srcRef, candIndex);
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
	combine(collectionIndex + 1, stack, qStack, collBegin + 1, collEnd, comps, names);
	stack.pop_back();
	qStack.pop_back();
      }
    }
  }
}

#endif
