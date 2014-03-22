#ifndef ParticleFlowReco_PFRecHitFwd_h
#define ParticleFlowReco_PFRecHitFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class PFRecHit;

  /// collection of PFRecHit objects
  typedef std::vector<PFRecHit> PFRecHitCollection;

  /// persistent reference to PFRecHit objects
  typedef edm::Ref<PFRecHitCollection> PFRecHitRef;

  /// reference to PFRecHit collection
  typedef edm::RefProd<PFRecHitCollection> PFRecHitRefProd;

  /// vector of references to PFRecHit objects all in the same collection
  typedef edm::RefVector<PFRecHitCollection> PFRecHitRefVector;

  /// iterator over a vector of references to PFRecHit objects
  typedef PFRecHitRefVector::iterator basicRecHit_iterator;

  /// ref to base vector for dealing with views
  typedef edm::RefToBaseVector<reco::PFRecHit> PFRecHitBaseRefVector;
}

#endif
