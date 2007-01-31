/* \class PtMinTrackCountFilter
 *
 * Filters events if at least N tracks above 
 * a pt cut are present
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"

typedef ObjectCountFilter<
          reco::TrackCollection, 
          PtMinSelector<reco::Track>
        > PtMinTrackCountFilter;

DEFINE_FWK_MODULE( PtMinTrackCountFilter );
