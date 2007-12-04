/** \class reco::modules::RecoTrackSelector
 *
 * Filter to select tracks according to pt, rapidity, tip, lip, number of hits
 *
 * \author Giuseppe Cerati, INFN
 *
 *  $Date: 2007/06/01 08:06:02 $
 *  $Revision: 1.1 $
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/RecoAlgos/interface/RecoTrackSelector.h"

namespace reco {
  namespace modules {
    typedef SingleObjectSelector<reco::TrackCollection,::RecoTrackSelector> 
    RecoTrackSelector ;

    DEFINE_FWK_MODULE( RecoTrackSelector );
  }
}
