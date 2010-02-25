/** \class reco::modules::CosmicTrackingParticleSelector
 *
 *  Filter to select Cosmc TrackingParticles according to pt, rapidity, tip, lip, number of hits, pdgId at PCA
 *  
 *  \author Yanyan Gao, FNAL
 *
 *  $Date: 2009/09/02 22:39:13 $
 *  $Revision: 1.1 $
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/RecoAlgos/interface/CosmicTrackingParticleSelector.h"

namespace reco {
  typedef ObjectSelector<CosmicTrackingParticleSelector> CosmicTrackingParticleSelector;
    DEFINE_FWK_MODULE( CosmicTrackingParticleSelector );
}
