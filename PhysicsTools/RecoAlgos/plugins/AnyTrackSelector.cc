/* \class AnyTrackSelector
 *
 * selects any track (just for testing purpose)
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/AnySelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"

 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::TrackCollection, 
             AnySelector<reco::Track> 
           > 
         > AnyTrackSelector;

DEFINE_FWK_MODULE( AnyTrackSelector );
