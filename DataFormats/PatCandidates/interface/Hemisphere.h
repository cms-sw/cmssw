#ifndef DataFormats_PatCandidates_Hemisphere_h
#define DataFormats_PatCandidates_Hemisphere_h
// #include "DataFormats/PatCandidates/interface/PATObject.h"

#include "DataFormats/Candidate/interface/CompositePtrCandidate.h"

// Define typedefs for convenience
namespace pat {
  class Hemisphere;
  typedef std::vector<Hemisphere> HemisphereCollection;
  typedef edm::Ref<HemisphereCollection> HemisphereRef;
  typedef edm::RefVector<HemisphereCollection> HemisphereRefVector;
}  // namespace pat

namespace pat {

  class Hemisphere : public reco::CompositePtrCandidate {
  public:
    Hemisphere() {}
    Hemisphere(const Candidate::LorentzVector& p4) : CompositePtrCandidate(0, p4) {}
    ~Hemisphere() override {}
  };

}  // namespace pat

#endif
