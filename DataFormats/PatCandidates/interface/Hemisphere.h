#ifndef DataFormats_PatCandidates_Hemisphere_h
#define DataFormats_PatCandidates_Hemisphere_h
// #include "DataFormats/PatCandidates/interface/PATObject.h"

#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"

namespace pat {
  
  class Hemisphere : public reco::CompositeRefBaseCandidate {
  public:
    Hemisphere () {}
    Hemisphere (const Particle::LorentzVector& p4) :
    CompositeRefBaseCandidate(0,p4) {}
    virtual ~Hemisphere () {}
  };

  typedef std::vector<Hemisphere> HemisphereCollection;
}

#endif
