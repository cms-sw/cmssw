#ifndef DataFormats_ParticleFlowReco_PFTrajectoryPointFwd_h
#define DataFormats_ParticleFlowReco_PFTrajectoryPointFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class PFTrajectoryPoint;

  /// collection of PFTrajectoryPoint objects
  typedef std::vector<PFTrajectoryPoint> PFTrajectoryPointCollection;

  /// persistent reference to PFTrajectoryPoint objects
  typedef edm::Ref<PFTrajectoryPointCollection> PFTrajectoryPointRef;

  /// reference to PFTrajectoryPoint collection
  typedef edm::RefProd<PFTrajectoryPointCollection> PFTrajectoryPointRefProd;

  /// vector of references to PFTrajectoryPoint objects all in the same collection
  typedef edm::RefVector<PFTrajectoryPointCollection> PFTrajectoryPointRefVector;

  /// iterator over a vector of references to PFTrajectoryPoint objects
  typedef PFTrajectoryPointRefVector::iterator trajectoryPoint_iterator;
}

#endif
