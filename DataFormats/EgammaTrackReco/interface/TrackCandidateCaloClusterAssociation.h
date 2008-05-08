#ifndef TrackCandidateCaloClusterAssociation_h
#define TrackCandidateCaloClusterAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"



namespace reco {

  // association map
  typedef edm::ValueMap<reco::CaloClusterPtr> TrackCandidateCaloClusterPtrAssociation;
  typedef edm::ValueMap<reco::CaloClusterPtrVector> TrackCandidateCaloClusterPtrVectorAssociation;


}

#endif
