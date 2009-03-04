/** \class reco::RecoTrackSelector
 *
 * Filter to select tracks according to pt, rapidity, tip, lip, number of hits, chi2
 *
 * \author Giuseppe Cerati, INFN
 *
 *  $Date: 2008/02/21 13:07:33 $
 *  $Revision: 1.3 $
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/RecoAlgos/interface/TrackSelector.h"
#include "CommonTools/RecoAlgos/interface/RecoTrackSelector.h"

namespace reco {
  typedef ObjectSelector<RecoTrackSelector> RecoTrackSelector;
  DEFINE_FWK_MODULE(RecoTrackSelector);
}
