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
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"

typedef ObjectCountFilter<
          reco::TrackCollection, 
          PtMinSelector
        > PtMinTrackCountFilter;

DEFINE_FWK_MODULE( PtMinTrackCountFilter );
