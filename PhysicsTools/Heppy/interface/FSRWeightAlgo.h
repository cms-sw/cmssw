#ifndef PhysicsTools_Heppy_FSRWeightAlgo_h
#define PhysicsTools_Heppy_FSRWeightAlgo_h

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

namespace heppy {

class FSRWeightAlgo {

 public:
  FSRWeightAlgo() {}
  virtual ~FSRWeightAlgo() {}
  void addGenParticle(const reco::GenParticle& gen) {genParticles_.push_back(gen);}
  void clear() {genParticles_.clear();}
  double weight() const;
  
 private:
  double alphaRatio(double) const;
  
  std::vector< reco::GenParticle > genParticles_;
  };
}
#endif
