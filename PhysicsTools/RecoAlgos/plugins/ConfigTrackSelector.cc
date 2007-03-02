/* \class PtMinTrackSelector
 *
 * selects track above a minumum pt cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"

 typedef SingleObjectSelector<
           reco::TrackCollection, 
           StringCutObjectSelector<reco::Track> 
         > ConfigTrackSelector;

DEFINE_FWK_MODULE( ConfigTrackSelector );
