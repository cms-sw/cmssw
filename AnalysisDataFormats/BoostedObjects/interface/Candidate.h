#ifndef ANALYSISDATAFORMATS_BOOSTEDOBJECTS_CANDIDATE_HH
#define ANALYSISDATAFORMATS_BOOSTEDOBJECTS_CANDIDATE_HH

/*\
 * Simple class to hold generic objects
 */

#include <TLorentzVector.h>

namespace vlq{
  class Candidate ; 
  typedef std::vector<Candidate> CandidateCollection ; 
  class Candidate {
    public:
      Candidate () {}
      ~Candidate () {} 
      Candidate ( const TLorentzVector& p4 ) :
        p4_(p4) 
      {}
      Candidate (const vlq::Candidate& cand ) 
      { *this = cand ; }
      float getPt () const { return p4_.Pt() ; } 
      float getEta () const { return p4_.Eta() ; }
      float getPhi () const { return p4_.Phi() ; }
      float getEnergy () const { return p4_.Energy() ; } 
      float getMass () const { return p4_.Mag() ; } 
      TLorentzVector getP4() const { return p4_ ; }
      void setP4 ( const TLorentzVector& p4 ) { p4_ = p4 ; } 
    protected: 
      TLorentzVector p4_ ; 
  };
}

#endif 
