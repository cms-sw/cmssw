#ifndef TrackCandidateSuperClusterAssociation_h
#define TrackCandidateSuperClusterAssociation_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

namespace reco {

  // association map
  typedef edm::ValueMap<reco::SuperClusterRef> TrackCandidateSuperClusterAssociationCollection;

}  // namespace reco

#endif
