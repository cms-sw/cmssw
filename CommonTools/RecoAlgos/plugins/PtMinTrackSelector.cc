/* \class PtMinTrackSelector
 *
 * selects track above a minumum pt cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/PtMinSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

 typedef SingleObjectSelector<
           reco::TrackCollection, 
           PtMinSelector
         > PtMinTrackSelector;

DEFINE_FWK_MODULE( PtMinTrackSelector );
