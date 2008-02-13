/** \class reco::RecoTrackSelector
 *
 * Filter to select tracks according to pt, rapidity, tip, lip, number of hits, chi2
 *
 * \author Giuseppe Cerati, INFN
 *
 *  $Date: 2007/11/09 13:52:55 $
 *  $Revision: 1.1 $
 *
 */

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"

#include "PhysicsTools/RecoAlgos/interface/RecoTrackSelector.h"

namespace reco {
    typedef ObjectSelector<RecoTrackSelector> RecoTrackSelector;
    DEFINE_FWK_MODULE( RecoTrackSelector );
}
