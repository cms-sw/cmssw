#ifndef AnalysisDataFormats_EWK_TwoObjectRefBaseCandidate_h
#define AnalysisDataFormats_EWK_TwoObjectRefBaseCandidate_h

#include <map>
#include <memory>

#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Ref.h"

namespace reco{
  
  template<typename t1,typename t2>
  class TwoObjectRefBaseCandidate : public reco::CompositeRefBaseCandidate
  {
  public:
    
    TwoObjectRefBaseCandidate() {}
    TwoObjectRefBaseCandidate(const reco::CandidateBaseRef&, const reco::CandidateBaseRef&);
    TwoObjectRefBaseCandidate(const edm::Ptr<t1>&, const edm::Ptr<t2>&);
    
    virtual ~TwoObjectRefBaseCandidate() {}

    const edm::Ptr<t1> daughter1Ptr() const; 
    const edm::Ptr<t2> daughter2Ptr() const; 
    
    const t1 daughter1() const;
    const t2 daughter2() const;        
  };
  
  template<typename t1,typename t2>
    TwoObjectRefBaseCandidate<t1,t2>::TwoObjectRefBaseCandidate(const reco::CandidateBaseRef& d1,
								const reco::CandidateBaseRef& d2)
    {
      addDaughter(d1);
      addDaughter(d2);
    }

  template<typename t1,typename t2>
    TwoObjectRefBaseCandidate<t1,t2>::TwoObjectRefBaseCandidate(const edm::Ptr<t1>& d1,
								const edm::Ptr<t2>& d2)
    {
      addDaughter(reco::CandidateBaseRef(d1));
      addDaughter(reco::CandidateBaseRef(d2));
    }
  
  template<typename t1,typename t2>
    const edm::Ptr<t1> 
    TwoObjectRefBaseCandidate<t1,t2>::daughter1Ptr() const 
    {
      return daughterRef(0).castTo<edm::Ptr<t1> >();
    }
  
  template<typename t1,typename t2>
    const edm::Ptr<t2>
    TwoObjectRefBaseCandidate<t1,t2>::daughter2Ptr() const 
    {
      return daughterRef(1).castTo<edm::Ptr<t2> >();
    }

  template<typename t1,typename t2>
    const t1
    TwoObjectRefBaseCandidate<t1,t2>::daughter1() const
    {
      edm::Ptr<t1> d = daughter1Ptr();
      if(d.isNonnull()) return *d;
      return t1();
    }
  
  template<typename t1,typename t2>
    const t2
    TwoObjectRefBaseCandidate<t1,t2>::daughter2() const
    {
      edm::Ptr<t2> d = daughter2Ptr();
      if(d.isNonnull()) return *d;
      return t2();
    }
  
}
#endif
