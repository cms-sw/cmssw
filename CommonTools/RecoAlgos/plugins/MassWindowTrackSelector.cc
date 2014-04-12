/* \class MassWindowTrackSelector
 *
 * selects the N track with largest pt
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/ObjectPairCollectionSelector.h"
#include "CommonTools/UtilAlgos/interface/RangeObjectPairSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "CommonTools/UtilAlgos/interface/MasslessInvariantMass.h"

 typedef ObjectSelector<
           ObjectPairCollectionSelector<
             reco::TrackCollection, 
             RangeObjectPairSelector<
               MasslessInvariantMass
             >
           >
         > MassWindowTrackSelector;

DEFINE_FWK_MODULE( MassWindowTrackSelector );
