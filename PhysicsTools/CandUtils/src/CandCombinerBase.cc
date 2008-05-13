#include "PhysicsTools/CandUtils/interface/CandCombinerBase.h"
#include <utility>
using namespace std;
using namespace reco;

CandCombinerBase::CandCombinerBase() :
  checkCharge_(false), dauCharge_(), overlap_() {
}

CandCombinerBase::CandCombinerBase(int q1, int q2) :
  checkCharge_(true), dauCharge_(2), overlap_() {
  dauCharge_[0] = q1;
  dauCharge_[1] = q2;
}

CandCombinerBase::CandCombinerBase(int q1, int q2, int q3) :
  checkCharge_(true), dauCharge_(3), overlap_() {
  dauCharge_[0] = q1;
  dauCharge_[1] = q2;
  dauCharge_[2] = q3;
}

CandCombinerBase::CandCombinerBase(int q1, int q2, int q3, int q4) :
  checkCharge_(true), dauCharge_(4), overlap_() {
  dauCharge_[0] = q1;
  dauCharge_[1] = q2;
  dauCharge_[2] = q3;
  dauCharge_[3] = q4;
}

CandCombinerBase::CandCombinerBase(bool checkCharge, const vector<int> & dauCharge) :
  checkCharge_(checkCharge), dauCharge_(dauCharge), overlap_() {
}

CandCombinerBase::~CandCombinerBase() {
}

bool CandCombinerBase::preselect(const Candidate & c1, const Candidate & c2) const {
  if (checkCharge_) {
    int dq1 = dauCharge_[0], dq2 = dauCharge_[1], q1 = c1.charge(), q2 = c2.charge();
    bool matchCharge = (q1 == dq1 && q2 == dq2) || (q1 == -dq1 && q2 == -dq2); 
    if (!matchCharge) return false; 
  }
  if (overlap_(c1, c2)) return false;
  return selectPair(c1, c2);
}

void CandCombinerBase::combine(CompositeCandidate & cmp, const CandidateBaseRef & c1, const CandidateBaseRef & c2) const {
  addDaughter(cmp, c1);
  addDaughter(cmp, c2);
  setup(cmp);
}

auto_ptr<CompositeCandidateCollection> 
CandCombinerBase::combine(const vector<edm::Handle<CandidateView> > & src) const {
  size_t srcSize = src.size();
  if (checkCharge_ && dauCharge_.size() != srcSize)
    throw edm::Exception(edm::errors::Configuration) 
      << "CandCombiner: trying to combine " << srcSize << " collections"
      << " but configured to check against " << dauCharge_.size() << " charges.";
  
  auto_ptr<CompositeCandidateCollection> comps(new CompositeCandidateCollection);
  if(srcSize == 2) {
    edm::Handle<CandidateView> src1 = src[0], src2 = src[1];
    if(src1.id() == src2.id()) {
      const CandidateView & cands = * src1;
      const int n = cands.size();
      for(int i1 = 0; i1 < n; ++i1) {
	const Candidate & c1 = cands[i1];
	CandidateBaseRef cr1(src1, i1);
	for(int i2 = i1 + 1; i2 < n; ++i2) {
	  const Candidate & c2 = cands[i2];
	  if (preselect(c1, c2)) {
	    CandidateBaseRef cr2(src2, i2);
	    CompositeCandidate c; 
	    combine(c, cr1, cr2);
	    if(select(c))
	      comps->push_back(c);
	  }
	}
      }
    } else {
      const CandidateView & cands1 = * src1, & cands2 = * src2;
      const int n1 = cands1.size(), n2 = cands2.size();
      for(int i1 = 0; i1 < n1; ++i1) {
	const Candidate & c1 = cands1[i1];
	CandidateBaseRef cr1(src1, i1);
	for(int i2 = 0; i2 < n2; ++i2) {
	  const Candidate & c2 = cands2[i2];
	  if(preselect(c1, c2)) {
	    CandidateBaseRef cr2(src2, i2);
	    CompositeCandidate c;
	    combine(c, cr1, cr2);
	    if(select(c))
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

auto_ptr<CompositeCandidateCollection> 
CandCombinerBase::combine(const edm::Handle<CandidateView> & src) const {
  if(checkCharge_ && dauCharge_.size() != 2)
    throw edm::Exception(edm::errors::Configuration) 
      << "CandCombiner: trying to combine 2 collections"
      << " but configured to check against " << dauCharge_.size() << " charges.";

  auto_ptr<CompositeCandidateCollection> comps(new CompositeCandidateCollection);
  const CandidateView & cands = * src; 
  const int n = cands.size();
  for(int i1 = 0; i1 < n; ++i1) {
    const Candidate & c1 = cands[i1];
    CandidateBaseRef cr1(src, i1);
    for(int i2 = i1 + 1; i2 < n; ++i2) {
      const Candidate & c2 = cands[i2];
      if(preselect(c1, c2)) {
	CandidateBaseRef cr2(src, i2);
	CompositeCandidate c;
	combine(c, cr1, cr2);
	if(select(c))
	  comps->push_back(c);
      }
    } 
  }

  return comps;
}

auto_ptr<CompositeCandidateCollection> 
CandCombinerBase::combine(const edm::Handle<CandidateView> & src1, const edm::Handle<CandidateView> & src2) const {
  vector<edm::Handle<CandidateView> > src;
  src.push_back(src1);
  src.push_back(src2);
  return combine(src);
}

auto_ptr<CompositeCandidateCollection> 
CandCombinerBase::combine(const edm::Handle<CandidateView> & src1, const edm::Handle<CandidateView> & src2, const edm::Handle<CandidateView> & src3) const {
  vector<edm::Handle<CandidateView> > src;
  src.push_back(src1);
  src.push_back(src2);
  src.push_back(src3);
  return combine(src);
}

auto_ptr<CompositeCandidateCollection> 
CandCombinerBase::combine(const edm::Handle<CandidateView> & src1, const edm::Handle<CandidateView> & src2, 
			  const edm::Handle<CandidateView> & src3, const edm::Handle<CandidateView> & src4) const {
  vector<edm::Handle<CandidateView> > src;
  src.push_back(src1);
  src.push_back(src2);
  src.push_back(src3);
  src.push_back(src4);
  return combine(src);
}

void CandCombinerBase::combine(size_t collectionIndex, CandStack & stack, ChargeStack & qStack,
			       vector<edm::Handle<CandidateView> >::const_iterator collBegin,
			       vector<edm::Handle<CandidateView> >::const_iterator collEnd,
			       auto_ptr<CompositeCandidateCollection> & comps) const {
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
      CompositeCandidate c;
      for(CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i) {
	addDaughter(c, i->first.first);
      }
      setup(c);
      if(select(c))
	comps->push_back(c);
    }
  } else {
    const edm::Handle<CandidateView> & srcRef = * collBegin;
    const CandidateView & src = * srcRef;
    size_t candBegin = 0, candEnd = src.size();
    for(CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i) 
      if(srcRef.id() == i->second->id()) 
	candBegin = i->first.second + 1;
    for(size_t candIndex = candBegin; candIndex != candEnd; ++ candIndex) {
      CandidateBaseRef candRef(srcRef, candIndex);
      bool noOverlap = true;
      const Candidate & cand = * candRef;
      for(CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i) 
	if(overlap_(cand, *(i->first.first))) { 
	  noOverlap = false; 
	  break; 
	}
      if(noOverlap) {
	stack.push_back(make_pair(make_pair(candRef, candIndex), collBegin));
	if(checkCharge_) qStack.push_back(cand.charge()); 
	combine(collectionIndex + 1, stack, qStack, collBegin + 1, collEnd, comps);
	stack.pop_back();
	qStack.pop_back();
      }
    }
  }
}
