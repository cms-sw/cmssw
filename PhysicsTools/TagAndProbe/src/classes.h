#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "PhysicsTools/TagAndProbe/interface/CandidateAssociation.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace
{
   struct dictionary
   {

     reco::CandViewCandViewAssociation a1;
     reco::CandViewCandViewAssociation::const_iterator it1;
     edm::Wrapper< reco::CandViewCandViewAssociation > w1;
     edm::helpers::KeyVal< edm::View< reco::Candidate >, edm::View< reco::Candidate > > k1;
     
     std::pair<edm::RefToBaseProd<reco::Candidate>,double> a;
     std::pair<edm::RefToBase<reco::Candidate>,double> aa;
     edm::Wrapper<edm::RefVector<std::vector<reco::RecoChargedCandidate>,reco::RecoChargedCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::RecoChargedCandidate>,reco::RecoChargedCandidate> > > aaa;
       
   };
}
