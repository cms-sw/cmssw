/** \class reco::RecoTrackSelector
 *
 * Filter to select tracks according to pt, rapidity, tip, lip, number of hits, chi2
 *
 * \author Giuseppe Cerati, INFN
 *
 *  $Date: 2013/06/04 21:29:52 $
 *  $Revision: 1.1.12.1 $
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "CommonTools/RecoAlgos/interface/TrackSelector.h"
#include "CommonTools/RecoAlgos/interface/TrackFullCloneSelectorBase.h"
#include "CommonTools/RecoAlgos/interface/RecoTrackSelector.h"

namespace reco {
  namespace modules {
    //typedef ObjectSelector<RecoTrackSelector> RecoTrackSelector;
    typedef TrackFullCloneSelectorBase<RecoTrackSelector> RecoTrackSelector;
    DEFINE_FWK_MODULE(RecoTrackSelector);
  }
}
