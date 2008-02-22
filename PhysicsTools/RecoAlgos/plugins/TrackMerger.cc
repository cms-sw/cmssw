/* \class TrackMerger
 *
 * merges track collections
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "PhysicsTools/UtilAlgos/interface/Merger.h"

typedef Merger<reco::TrackCollection> TrackMerger;

DEFINE_FWK_MODULE( TrackMerger );
