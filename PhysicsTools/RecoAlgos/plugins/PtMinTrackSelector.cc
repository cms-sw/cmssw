/* \class PtMinTrackSelector
 *
 * selects track above a minumum pt cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"

 typedef SingleObjectSelector<
           reco::TrackCollection, 
           PtMinSelector<reco::Track> 
         > PtMinTrackSelector;

DEFINE_FWK_MODULE( PtMinTrackSelector );
