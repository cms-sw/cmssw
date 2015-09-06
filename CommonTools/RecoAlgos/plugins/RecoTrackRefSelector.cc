/** \class reco::RecoTrackRefSelector
 *
 * Filter to select tracks according to pt, rapidity, tip, lip, number of hits, chi2
 *
 * \author Giuseppe Cerati, INFN
 * \author Ian Tomalin, RAL
 *
 *  $Date: 2009/07/09 13:11:30 $
 *  $Revision: 1.1 $
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"
//#include "CommonTools/RecoAlgos/interface/TrackSelector.h"
#include "CommonTools/RecoAlgos/interface/RecoTrackRefSelector.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
  typedef ObjectSelectorStream<RecoTrackRefSelector,reco::TrackRefVector> RecoTrackRefSelector;
  DEFINE_FWK_MODULE(RecoTrackRefSelector);
}
