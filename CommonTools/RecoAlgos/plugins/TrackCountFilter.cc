/* \class TrackCountFilter
 *
 * Filters events if at least N tracks
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"

 typedef ObjectCountFilter<
           reco::TrackCollection
         >::type TrackCountFilter;

DEFINE_FWK_MODULE( TrackCountFilter );
