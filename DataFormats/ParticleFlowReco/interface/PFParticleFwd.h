#ifndef DataFormats_ParticleFlowReco_PFParticleFwd_h
#define DataFormats_ParticleFlowReco_PFParticleFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class PFParticle;

  /// collection of PFParticle objects
  typedef std::vector<PFParticle> PFParticleCollection;

  /// persistent reference to PFParticle objects
  typedef edm::Ref<PFParticleCollection> PFParticleRef;

  /// reference to PFParticle collection
  typedef edm::RefProd<PFParticleCollection> PFParticleRefProd;

  /// vector of references to PFParticle objects all in the same collection
  typedef edm::RefVector<PFParticleCollection> PFParticleRefVector;

  /// iterator over a vector of references to PFParticle objects
  typedef PFParticleRefVector::iterator pfParticle_iterator;
}

#endif
