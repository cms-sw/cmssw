#ifndef TrackCaloClusterAssociation_h
#define TrackCaloClusterAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"


namespace reco {

  // association map
  //  typedef edm::ValueMap<reco::CaloClusterPtrVector> TrackCaloClusterAssociationCollection;

  typedef edm::ValueMap<reco::CaloClusterPtr>       TrackCaloClusterPtrAssociation;
  typedef edm::ValueMap<reco::CaloClusterPtrVector> TrackCaloClusterPtrVectorAssociation;


}

#endif
