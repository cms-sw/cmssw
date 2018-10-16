#include "CommonTools/CandUtils/interface/NamedCandCombinerBase.h"
#include <utility>
using namespace std;
using namespace reco;

NamedCandCombinerBase::NamedCandCombinerBase(std::string name) :
  checkCharge_(false), checkOverlap_(true), dauCharge_(), overlap_(), name_(name) {
}

NamedCandCombinerBase::NamedCandCombinerBase(std::string name, int q1, int q2) :
  checkCharge_(true), checkOverlap_(true), dauCharge_(2), overlap_(), name_(name) {
  dauCharge_[0] = q1;
  dauCharge_[1] = q2;
}

NamedCandCombinerBase::NamedCandCombinerBase(std::string name, int q1, int q2, int q3) :
  checkCharge_(true), checkOverlap_(true), dauCharge_(3), overlap_(), name_(name) {
  dauCharge_[0] = q1;
  dauCharge_[1] = q2;
  dauCharge_[2] = q3;
}

NamedCandCombinerBase::NamedCandCombinerBase(std::string name, int q1, int q2, int q3, int q4) :
  checkCharge_(true), checkOverlap_(true), dauCharge_(4), overlap_(), name_(name) {
  dauCharge_[0] = q1;
  dauCharge_[1] = q2;
  dauCharge_[2] = q3;
  dauCharge_[3] = q4;
}

NamedCandCombinerBase::NamedCandCombinerBase(std::string name, bool checkCharge, bool checkOverlap, const vector<int> & dauCharge) :
  checkCharge_(checkCharge), checkOverlap_(checkOverlap), dauCharge_(dauCharge), overlap_() {
}

NamedCandCombinerBase::~NamedCandCombinerBase() {
}

bool NamedCandCombinerBase::preselect(const Candidate & c1, const Candidate & c2) const {
  if (checkCharge_) {
    int dq1 = dauCharge_[0], dq2 = dauCharge_[1], q1 = c1.charge(), q2 = c2.charge();
    bool matchCharge = (q1 == dq1 && q2 == dq2) || (q1 == -dq1 && q2 == -dq2); 
    if (!matchCharge) return false; 
  }
  if (checkOverlap_ && overlap_(c1, c2)) return false;
  return selectPair(c1, c2);
}

void NamedCandCombinerBase::combine(NamedCompositeCandidate & cmp, const CandidatePtr & c1, const CandidatePtr & c2,
				    std::string n1, std::string n2 ) const {
  addDaughter(cmp, c1, n1);
  addDaughter(cmp, c2, n2);
  setup(cmp);
}

unique_ptr<NamedCompositeCandidateCollection> 
NamedCandCombinerBase::combine(const vector<CandidatePtrVector> & src,
			       string_coll const & names) const {
  size_t srcSize = src.size();
  if (checkCharge_ && dauCharge_.size() != srcSize)
    throw edm::Exception(edm::errors::Configuration) 
      << "NamedCandCombiner: trying to combine " << srcSize << " collections"
      << " but configured to check against " << dauCharge_.size() << " charges.";

  if ( names.size() < 2 )
    throw edm::Exception(edm::errors::Configuration)
      << "NamedCandCombiner: need to add 2 names, but size is " << names.size();
  
  unique_ptr<NamedCompositeCandidateCollection> comps(new NamedCompositeCandidateCollection);
  if(srcSize == 2) {
    CandidatePtrVector src1 = src[0], src2 = src[1];
    if(src1 == src2) {
      const int n = src1.size();
      for(int i1 = 0; i1 < n; ++i1) {
	const Candidate & c1 = *(src1[i1]);
	for(int i2 = i1 + 1; i2 < n; ++i2) {
	  const Candidate & c2 = *(src1[i2]);
	  if (preselect(c1, c2)) {
	    NamedCompositeCandidate c; 
	    combine(c, src1[i1], src1[i2], names[0], names[1]);
	    if(select(c))
	      comps->push_back(c);
	  }
	}
      }
    } else {
      const int n1 = src1.size(), n2 = src2.size();
      for(int i1 = 0; i1 < n1; ++i1) {
	const Candidate & c1 = *(src1[i1]);
	for(int i2 = 0; i2 < n2; ++i2) {
	  const Candidate & c2 = *(src2[i2]);
	  if(preselect(c1, c2)) {
	    NamedCompositeCandidate c;
	    combine(c, src1[i1], src2[i2], names[0], names[1]);
	    if(select(c))
	      comps->push_back(c);
	  }
	}
      }
    }
  } else {
    CandStack stack;
    ChargeStack qStack;
    combine(0, stack, qStack, names, src.begin(), src.end(), comps);
  }

  return std::move(comps);
}

unique_ptr<NamedCompositeCandidateCollection> 
NamedCandCombinerBase::combine(const CandidatePtrVector & src, string_coll const & names) const {
  if(checkCharge_ && dauCharge_.size() != 2)
    throw edm::Exception(edm::errors::Configuration) 
      << "NamedCandCombiner: trying to combine 2 collections"
      << " but configured to check against " << dauCharge_.size() << " charges.";

  if ( names.size() < 2 )
    throw edm::Exception(edm::errors::Configuration)
      << "NamedCandCombiner: need to add 2 names, but size is " << names.size();

  unique_ptr<NamedCompositeCandidateCollection> comps(new NamedCompositeCandidateCollection);
  const int n = src.size();
  for(int i1 = 0; i1 < n; ++i1) {
    const Candidate & c1 = *(src[i1]);
    for(int i2 = i1 + 1; i2 < n; ++i2) {
      const Candidate & c2 = *(src[i2]);
      if(preselect(c1, c2)) {
	NamedCompositeCandidate c;
	combine(c, src[i1], src[i2], names[0], names[1]);
	if(select(c))
	  comps->push_back(c);
      }
    } 
  }

  return std::move(comps);
}

unique_ptr<NamedCompositeCandidateCollection> 
NamedCandCombinerBase::combine(const CandidatePtrVector & src1, const CandidatePtrVector & src2, string_coll const & names) const {
  vector<CandidatePtrVector> src;
  src.push_back(src1);
  src.push_back(src2);
  return std::move(combine(src, names));
}

unique_ptr<NamedCompositeCandidateCollection> 
NamedCandCombinerBase::combine(const CandidatePtrVector & src1, const CandidatePtrVector & src2, const CandidatePtrVector & src3, 
			       string_coll const & names) const {
  vector<CandidatePtrVector> src;
  src.push_back(src1);
  src.push_back(src2);
  src.push_back(src3);
  return std::move(combine(src, names));
}

unique_ptr<NamedCompositeCandidateCollection> 
NamedCandCombinerBase::combine(const CandidatePtrVector & src1, const CandidatePtrVector & src2, 
			       const CandidatePtrVector & src3, const CandidatePtrVector & src4, 
			       string_coll const & names) const {
  vector<CandidatePtrVector> src;
  src.push_back(src1);
  src.push_back(src2);
  src.push_back(src3);
  src.push_back(src4);
  return std::move(combine(src, names));
}

void NamedCandCombinerBase::combine(size_t collectionIndex, CandStack & stack, ChargeStack & qStack, 
				    string_coll const & names,
				    vector<CandidatePtrVector>::const_iterator collBegin,
				    vector<CandidatePtrVector>::const_iterator collEnd,
				    unique_ptr<NamedCompositeCandidateCollection> & comps) const {
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
      NamedCompositeCandidate c;
      int ii = 0;
      for(CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i, ++ii) {
	addDaughter(c, i->first.first, names[ii]);
      }
      setup(c);
      if(select(c))
	comps->push_back(c);
    }
  } else {
    const CandidatePtrVector & src = * collBegin;
    size_t candBegin = 0, candEnd = src.size();
    for(CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i) 
      if(src == * i->second) 
	candBegin = i->first.second + 1;
    for(size_t candIndex = candBegin; candIndex != candEnd; ++ candIndex) {
      const CandidatePtr & candPtr(src[candIndex]);

      bool noOverlap = true;
      const Candidate & cand = *candPtr;
      for(CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i) 
	if(checkOverlap_ && overlap_(cand, *(i->first.first))) { 
	  noOverlap = false; 
	  break; 
	}
      if(noOverlap) {
	stack.push_back(make_pair(make_pair(candPtr, candIndex), collBegin));
	if(checkCharge_) qStack.push_back(cand.charge()); 
	combine(collectionIndex + 1, stack, qStack, names, collBegin + 1, collEnd, comps);
	stack.pop_back();
	qStack.pop_back();
      }
    }
  }
}
