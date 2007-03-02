/* \class PtMinTrackSelector
 *
 * selects track above a minumum pt cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"

 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::TrackCollection, 
             StringCutObjectSelector<reco::Track> 
           >
         > ConfigTrackSelector;

DEFINE_FWK_MODULE( ConfigTrackSelector );
