#ifndef TrackSuperClusterAssociation_h
#define TrackSuperClusterAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"


namespace reco {

  // association map
  typedef edm::ValueMap<reco::SuperClusterRef> TrackSuperClusterAssociationCollection;

}

#endif
