#ifndef CandUtils_makeNamedCompositeCandidate_h
#define CandUtils_makeNamedCompositeCandidate_h
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include <memory>
#include <string>

namespace helpers {
  struct NamedCompositeCandidateMaker {
    NamedCompositeCandidateMaker( std::auto_ptr<reco::NamedCompositeCandidate> cmp ) :
      cmp_( cmp ) {
    }
    void addDaughter( const reco::Candidate & dau, std::string name ) {
      cmp_->addDaughter( dau, name );
    }
    template<typename S>
    std::auto_ptr<reco::Candidate> operator[]( const S & setup ) {
      setup.set( * cmp_ );
      return release();
    }
  private:
    std::auto_ptr<reco::NamedCompositeCandidate> cmp_;
    std::auto_ptr<reco::Candidate> release() {
      std::auto_ptr<reco::Candidate> ret( cmp_.get() );
      cmp_.release();
      return ret;
    }
  };
}

helpers::NamedCompositeCandidateMaker 
makeNamedCompositeCandidate( const reco::Candidate & c1, std::string s1,
			     const reco::Candidate & c2, std::string s2 );

helpers::NamedCompositeCandidateMaker 
makeNamedCompositeCandidate( const reco::Candidate & c1, std::string s1,
			     const reco::Candidate & c2, std::string s2,
			     const reco::Candidate & c3, std::string s3 );

helpers::NamedCompositeCandidateMaker 
makeNamedCompositeCandidate( const reco::Candidate & c1, std::string s1,
			     const reco::Candidate & c2, std::string s2,
			     const reco::Candidate & c3, std::string s3 );


helpers::NamedCompositeCandidateMaker 
makeNamedCompositeCandidate( const reco::Candidate & c1, std::string s1,
			     const reco::Candidate & c2, std::string s2,
			     const reco::Candidate & c3, std::string s3,
			     const reco::Candidate & c4, std::string s4 );

template<typename C>
helpers::NamedCompositeCandidateMaker 
makeNamedCompositeCandidate( const typename C::const_iterator & begin, 
			     const typename C::const_iterator & end,
			     const std::vector<std::string>::const_iterator sbegin,
			     const std::vector<std::string>::const_iterator send ) {
  helpers::NamedCompositeCandidateMaker 
    cmp( std::auto_ptr<reco::NamedCompositeCandidate>( new reco::NamedCompositeCandidate ) );
  std::vector<std::string>::const_iterator si = sbegin;
  for( typename C::const_iterator i = begin; i != end && si != send; ++ i, ++si ) 
    cmp.addDaughter( * i, * si );
  return cmp;
}

helpers::NamedCompositeCandidateMaker 
makeNamedCompositeCandidateWithRefsToMaster( const reco::CandidateRef & c1, std::string s1, 
					     const reco::CandidateRef & c2, std::string s2 );

helpers::NamedCompositeCandidateMaker 
makeNamedCompositeCandidateWithRefsToMaster( const reco::CandidateRef & c1, std::string s1, 
					     const reco::CandidateRef & c2, std::string s2,
					     const reco::CandidateRef & c3, std::string s3 );

helpers::NamedCompositeCandidateMaker 
makeNamedCompositeCandidateWithRefsToMaster( const reco::CandidateRef & c1, std::string s1, 
					     const reco::CandidateRef & c2, std::string s2,
					     const reco::CandidateRef & c3, std::string s3,
					     const reco::CandidateRef & c4, std::string s4 );

template<typename C>
helpers::NamedCompositeCandidateMaker 
makeNamedCompositeCandidateWithRefsToMaster( const typename C::const_iterator & begin, 
					     const typename C::const_iterator & end,
			     const std::vector<std::string>::const_iterator sbegin,
			     const std::vector<std::string>::const_iterator send ) {
  helpers::NamedCompositeCandidateMaker 
    cmp( std::auto_ptr<reco::NamedCompositeCandidate>( new reco::NamedCompositeCandidate ) );
  std::vector<std::string>::const_iterator si = sbegin;
  for( typename C::const_iterator i = begin; i != end && si != send; ++ i, ++ si ) 
    cmp.addDaughter( ShallowCloneCandidate( CandidateBaseRef( * i ) ), * si );
  return cmp;
}

#endif
