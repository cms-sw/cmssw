/* \class MassWindowTrackSelector
 *
 * selects the N track with largest pt
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectPairCollectionSelector.h"
#include "PhysicsTools/UtilAlgos/interface/RangeObjectPairSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "PhysicsTools/UtilAlgos/interface/MasslessInvariantMass.h"

 typedef ObjectSelector<
           ObjectPairCollectionSelector<
             reco::TrackCollection, 
             RangeObjectPairSelector<
               reco::Track,
               MasslessInvariantMass<reco::Track>
             >
           >
         > MassWindowTrackSelector;

DEFINE_FWK_MODULE( MassWindowTrackSelector );
