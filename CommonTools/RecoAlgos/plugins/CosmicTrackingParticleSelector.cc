/** \class reco::modules::CosmicTrackingParticleSelector
 *
 *  Filter to select Cosmc TrackingParticles according to pt, rapidity, tip, lip, number of hits, pdgId at PCA
 *  
 *  \author Yanyan Gao, FNAL
 *
 *  $Date: 2009/07/15 13:11:31 $
 *  $Revision: 1.1 $
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/RecoAlgos/interface/CosmicTrackingParticleSelector.h"

namespace reco {
  typedef ObjectSelector<CosmicTrackingParticleSelector> CosmicTrackingParticleSelector;
    DEFINE_ANOTHER_FWK_MODULE( CosmicTrackingParticleSelector );
}
