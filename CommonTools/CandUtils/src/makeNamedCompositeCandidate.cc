#include <utility>

#include "CommonTools/CandUtils/interface/makeNamedCompositeCandidate.h"
using namespace reco;
using namespace std;

helpers::NamedCompositeCandidateMaker makeNamedCompositeCandidate(const Candidate& c1,
                                                                  std::string s1,
                                                                  const Candidate& c2,
                                                                  std::string s2) {
  helpers::NamedCompositeCandidateMaker cmp(unique_ptr<NamedCompositeCandidate>(new NamedCompositeCandidate));
  cmp.addDaughter(c1, std::move(s1));
  cmp.addDaughter(c2, std::move(s2));
  return cmp;
}

helpers::NamedCompositeCandidateMaker makeNamedCompositeCandidate(
    const Candidate& c1, std::string s1, const Candidate& c2, std::string s2, const Candidate& c3, std::string s3) {
  helpers::NamedCompositeCandidateMaker cmp(unique_ptr<NamedCompositeCandidate>(new NamedCompositeCandidate));
  cmp.addDaughter(c1, std::move(s1));
  cmp.addDaughter(c2, std::move(s2));
  cmp.addDaughter(c3, std::move(s3));
  return cmp;
}

helpers::NamedCompositeCandidateMaker makeNamedCompositeCandidate(const Candidate& c1,
                                                                  std::string s1,
                                                                  const Candidate& c2,
                                                                  std::string s2,
                                                                  const Candidate& c3,
                                                                  std::string s3,
                                                                  const Candidate& c4,
                                                                  std::string s4) {
  helpers::NamedCompositeCandidateMaker cmp(unique_ptr<NamedCompositeCandidate>(new NamedCompositeCandidate));
  cmp.addDaughter(c1, std::move(s1));
  cmp.addDaughter(c2, std::move(s2));
  cmp.addDaughter(c3, std::move(s3));
  cmp.addDaughter(c4, std::move(s4));
  return cmp;
}

helpers::NamedCompositeCandidateMaker makeNamedCompositeCandidateWithRefsToMaster(const reco::CandidateRef& c1,
                                                                                  std::string s1,
                                                                                  const reco::CandidateRef& c2,
                                                                                  std::string s2) {
  helpers::NamedCompositeCandidateMaker cmp(unique_ptr<NamedCompositeCandidate>(new NamedCompositeCandidate));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c1)), std::move(s1));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c2)), std::move(s2));
  return cmp;
}

helpers::NamedCompositeCandidateMaker makeNamedCompositeCandidateWithRefsToMaster(const reco::CandidateRef& c1,
                                                                                  std::string s1,
                                                                                  const reco::CandidateRef& c2,
                                                                                  std::string s2,
                                                                                  const reco::CandidateRef& c3,
                                                                                  std::string s3) {
  helpers::NamedCompositeCandidateMaker cmp(unique_ptr<NamedCompositeCandidate>(new NamedCompositeCandidate));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c1)), std::move(s1));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c2)), std::move(s2));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c3)), std::move(s3));
  return cmp;
}

helpers::NamedCompositeCandidateMaker makeNamedCompositeCandidateWithRefsToMaster(const reco::CandidateRef& c1,
                                                                                  std::string s1,
                                                                                  const reco::CandidateRef& c2,
                                                                                  std::string s2,
                                                                                  const reco::CandidateRef& c3,
                                                                                  std::string s3,
                                                                                  const reco::CandidateRef& c4,
                                                                                  std::string s4) {
  helpers::NamedCompositeCandidateMaker cmp(unique_ptr<NamedCompositeCandidate>(new NamedCompositeCandidate));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c1)), std::move(s1));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c2)), std::move(s2));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c3)), std::move(s3));
  cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(c4)), std::move(s4));
  return cmp;
}
