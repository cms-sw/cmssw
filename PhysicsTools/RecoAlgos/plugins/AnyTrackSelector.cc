/* \class AnyTrackSelector
 *
 * selects any track (just for testing purpose)
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/AnySelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

 typedef SingleObjectSelector<
           reco::TrackCollection, 
           AnySelector
         > AnyTrackSelector;

DEFINE_FWK_MODULE( AnyTrackSelector );
