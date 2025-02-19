#ifndef CandUtils_makeCompositeCandidate_h
#define CandUtils_makeCompositeCandidate_h
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositePtrCandidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include <memory>

namespace helpers {
  struct CompositeCandidateMaker {
    CompositeCandidateMaker(std::auto_ptr<reco::CompositeCandidate> cmp) :
      cmp_( cmp ) {
    }
    void addDaughter(const reco::Candidate & dau) {
      cmp_->addDaughter( dau );
    }
    template<typename S>
    std::auto_ptr<reco::Candidate> operator[]( const S & setup ) {
      setup.set( * cmp_ );
      return release();
    }
  private:
    std::auto_ptr<reco::CompositeCandidate> cmp_;
    std::auto_ptr<reco::Candidate> release() {
      std::auto_ptr<reco::Candidate> ret( cmp_.get() );
      cmp_.release();
      return ret;
    }
  };

  struct CompositePtrCandidateMaker {
    CompositePtrCandidateMaker(std::auto_ptr<reco::CompositePtrCandidate> cmp) :
      cmp_( cmp ) {
    }
    void addDaughter(const reco::CandidatePtr & dau) {
      cmp_->addDaughter( dau );
    }
    template<typename S>
    std::auto_ptr<reco::Candidate> operator[]( const S & setup ) {
      setup.set( * cmp_ );
      return release();
    }
  private:
    std::auto_ptr<reco::CompositePtrCandidate> cmp_;
    std::auto_ptr<reco::Candidate> release() {
      std::auto_ptr<reco::Candidate> ret( cmp_.get() );
      cmp_.release();
      return ret;
    }
  };
}

helpers::CompositeCandidateMaker 
makeCompositeCandidate(const reco::Candidate & c1, 
		       const reco::Candidate & c2);

helpers::CompositeCandidateMaker 
makeCompositeCandidate(const reco::Candidate & c1, 
		       const reco::Candidate & c2,
		       const reco::Candidate & c3);

helpers::CompositeCandidateMaker 
makeCompositeCandidate(const reco::Candidate & c1, 
		       const reco::Candidate & c2,
		       const reco::Candidate & c3);


helpers::CompositeCandidateMaker 
makeCompositeCandidate(const reco::Candidate & c1, 
		       const reco::Candidate & c2,
		       const reco::Candidate & c3,
		       const reco::Candidate & c4);

template<typename C>
helpers::CompositeCandidateMaker 
makeCompositeCandidate(const typename C::const_iterator & begin, 
		       const typename C::const_iterator & end) {
  helpers::CompositeCandidateMaker 
    cmp(std::auto_ptr<reco::CompositeCandidate>(new reco::CompositeCandidate) );
  for(typename C::const_iterator i = begin; i != end; ++ i) 
    cmp.addDaughter(* i);
  return cmp;
}

helpers::CompositeCandidateMaker 
makeCompositeCandidateWithRefsToMaster(const reco::CandidateRef & c1, 
				       const reco::CandidateRef & c2);

helpers::CompositeCandidateMaker 
makeCompositeCandidateWithRefsToMaster(const reco::CandidateRef & c1, 
				       const reco::CandidateRef & c2,
				       const reco::CandidateRef & c3);

helpers::CompositeCandidateMaker 
makeCompositeCandidateWithRefsToMaster(const reco::CandidateRef & c1, 
				       const reco::CandidateRef & c2,
				       const reco::CandidateRef & c3,
				       const reco::CandidateRef & c4);

template<typename C>
helpers::CompositeCandidateMaker 
makeCompositeCandidateWithRefsToMaster(const typename C::const_iterator & begin, 
				       const typename C::const_iterator & end) {
  helpers::CompositeCandidateMaker 
    cmp(std::auto_ptr<reco::CompositeCandidate>(new reco::CompositeCandidate));
  for(typename C::const_iterator i = begin; i != end; ++ i) 
    cmp.addDaughter(ShallowCloneCandidate(CandidateBaseRef(* i)));
  return cmp;
}

helpers::CompositePtrCandidateMaker 
makeCompositePtrCandidate(const reco::CandidatePtr & c1, 
			  const reco::CandidatePtr & c2);

helpers::CompositePtrCandidateMaker 
makeCompositePtrCandidate(const reco::CandidatePtr & c1, 
			  const reco::CandidatePtr & c2,
			  const reco::CandidatePtr & c3);

helpers::CompositePtrCandidateMaker 
makeCompositePtrCandidate(const reco::CandidatePtr & c1, 
			  const reco::CandidatePtr & c2,
			  const reco::CandidatePtr & c3);


helpers::CompositePtrCandidateMaker 
makeCompositePtrCandidate(const reco::CandidatePtr & c1, 
			  const reco::CandidatePtr & c2,
			  const reco::CandidatePtr & c3,
			  const reco::CandidatePtr & c4);

#endif
