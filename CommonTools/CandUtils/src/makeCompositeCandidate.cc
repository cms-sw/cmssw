#include <memory>

#include "CommonTools/CandUtils/interface/makeCompositeCandidate.h"
using namespace reco;
using namespace std;

helpers::CompositeCandidateMaker makeCompositeCandidate(const Candidate& c1, const Candidate& c2) {
  helpers::CompositeCandidateMaker cmp(std::make_unique<CompositeCandidate>());
  cmp.addDaughter(c1);
  cmp.addDaughter(c2);
  return cmp;
}

helpers::CompositeCandidateMaker makeCompositeCandidate(const Candidate& c1, const Candidate& c2, const Candidate& c3) {
  helpers::CompositeCandidateMaker cmp(std::make_unique<CompositeCandidate>());
  cmp.addDaughter(c1);
  cmp.addDaughter(c2);
  cmp.addDaughter(c3);
  return cmp;
}

helpers::CompositeCandidateMaker makeCompositeCandidate(const Candidate& c1,
                                                        const Candidate& c2,
                                                        const Candidate& c3,
                                                        const Candidate& c4) {
  helpers::CompositeCandidateMaker cmp(std::make_unique<CompositeCandidate>());
  cmp.addDaughter(c1);
  cmp.addDaughter(c2);
  cmp.addDaughter(c3);
  cmp.addDaughter(c4);
  return cmp;
}

helpers::CompositeCandidateMaker makeCompositeCandidateWithRefsToMaster(const reco::CandidateRef& c1,
                                                                        const reco::CandidateRef& c2) {
  helpers::CompositeCandidateMaker cmp(std::make_unique<CompositeCandidate>());
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c1)));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c2)));
  return cmp;
}

helpers::CompositeCandidateMaker makeCompositeCandidateWithRefsToMaster(const reco::CandidateRef& c1,
                                                                        const reco::CandidateRef& c2,
                                                                        const reco::CandidateRef& c3) {
  helpers::CompositeCandidateMaker cmp(std::make_unique<CompositeCandidate>());
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c1)));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c2)));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c3)));
  return cmp;
}

helpers::CompositeCandidateMaker makeCompositeCandidateWithRefsToMaster(const reco::CandidateRef& c1,
                                                                        const reco::CandidateRef& c2,
                                                                        const reco::CandidateRef& c3,
                                                                        const reco::CandidateRef& c4) {
  helpers::CompositeCandidateMaker cmp(std::make_unique<CompositeCandidate>());
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c1)));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c2)));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c3)));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c4)));
  return cmp;
}

helpers::CompositePtrCandidateMaker makeCompositePtrCandidate(const CandidatePtr& c1, const CandidatePtr& c2) {
  helpers::CompositePtrCandidateMaker cmp(std::make_unique<CompositePtrCandidate>());
  cmp.addDaughter(c1);
  cmp.addDaughter(c2);
  return cmp;
}

helpers::CompositePtrCandidateMaker makeCompositePtrCandidate(const CandidatePtr& c1,
                                                              const CandidatePtr& c2,
                                                              const CandidatePtr& c3) {
  helpers::CompositePtrCandidateMaker cmp(std::make_unique<CompositePtrCandidate>());
  cmp.addDaughter(c1);
  cmp.addDaughter(c2);
  cmp.addDaughter(c3);
  return cmp;
}

helpers::CompositePtrCandidateMaker makeCompositePtrCandidate(const CandidatePtr& c1,
                                                              const CandidatePtr& c2,
                                                              const CandidatePtr& c3,
                                                              const CandidatePtr& c4) {
  helpers::CompositePtrCandidateMaker cmp(std::make_unique<CompositePtrCandidate>());
  cmp.addDaughter(c1);
  cmp.addDaughter(c2);
  cmp.addDaughter(c3);
  cmp.addDaughter(c4);
  return cmp;
}
