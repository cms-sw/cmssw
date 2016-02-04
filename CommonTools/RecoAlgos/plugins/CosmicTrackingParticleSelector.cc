/** \class reco::modules::CosmicTrackingParticleSelector
 *
 *  Filter to select Cosmc TrackingParticles according to pt, rapidity, tip, lip, number of hits, pdgId at PCA
 *  
 *  \author Yanyan Gao, FNAL
 *
 *  $Date: 2010/02/25 00:28:55 $
 *  $Revision: 1.2 $
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/RecoAlgos/interface/CosmicTrackingParticleSelector.h"

namespace reco {
  typedef ObjectSelector<CosmicTrackingParticleSelector> CosmicTrackingParticleSelector;
    DEFINE_FWK_MODULE( CosmicTrackingParticleSelector );
}
