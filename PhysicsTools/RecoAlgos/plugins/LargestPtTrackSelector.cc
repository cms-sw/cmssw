/* \class LargestPtTrackSelector
 *
 * selects the N track with largest pt
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SortCollectionSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "PhysicsTools/Utilities/interface/PtComparator.h"

 typedef ObjectSelector<
           SortCollectionSelector<
             reco::TrackCollection, 
             PtInverseComparator<reco::Track> 
           > 
         > LargestPtTrackSelector;

DEFINE_FWK_MODULE( LargestPtTrackSelector );
