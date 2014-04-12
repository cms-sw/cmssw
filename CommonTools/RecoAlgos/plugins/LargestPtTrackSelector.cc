/* \class LargestPtTrackSelector
 *
 * selects the N track with largest pt
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SortCollectionSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "CommonTools/Utils/interface/PtComparator.h"

 typedef ObjectSelector<
           SortCollectionSelector<
             reco::TrackCollection, 
             GreaterByPt<reco::Track> 
           > 
         > LargestPtTrackSelector;

DEFINE_FWK_MODULE( LargestPtTrackSelector );
