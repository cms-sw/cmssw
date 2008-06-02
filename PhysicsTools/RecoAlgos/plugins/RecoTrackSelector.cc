/** \class reco::RecoTrackSelector
 *
 * Filter to select tracks according to pt, rapidity, tip, lip, number of hits, chi2
 *
 * \author Giuseppe Cerati, INFN
 *
 *  $Date: 2008/02/13 10:39:56 $
 *  $Revision: 1.2 $
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"
#include "PhysicsTools/RecoAlgos/interface/RecoTrackSelector.h"

namespace reco {
  typedef ObjectSelector<RecoTrackSelector> RecoTrackSelector;
  DEFINE_FWK_MODULE(RecoTrackSelector);
}
