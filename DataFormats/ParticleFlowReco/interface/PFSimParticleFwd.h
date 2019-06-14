#ifndef DataFormats_ParticleFlowReco_PFSimParticleFwd_h
#define DataFormats_ParticleFlowReco_PFSimParticleFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class PFSimParticle;

  /// collection of PFSimParticle objects
  typedef std::vector<PFSimParticle> PFSimParticleCollection;

  /// persistent reference to PFSimParticle objects
  typedef edm::Ref<PFSimParticleCollection> PFSimParticleRef;

  /// reference to PFSimParticle collection
  typedef edm::RefProd<PFSimParticleCollection> PFSimParticleRefProd;

  /// vector of references to PFSimParticle objects all in the same collection
  typedef edm::RefVector<PFSimParticleCollection> PFSimParticleRefVector;

  /// iterator over a vector of references to PFSimParticle objects
  typedef PFSimParticleRefVector::iterator pfParticle_iterator;
}  // namespace reco

#endif
