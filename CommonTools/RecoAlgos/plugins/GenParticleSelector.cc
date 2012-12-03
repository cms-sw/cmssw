/** \class reco::modules::GenParticleSelector
 *
 *  Filter to select GenParticles
 *
 *  \author Giuseppe Cerati, UCSD
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/RecoAlgos/interface/GenParticleSelector.h"

namespace reco {
  namespace modules {
    typedef SingleObjectSelector<GenParticleCollection,::GenParticleSelector> 
    GenParticleSelector ;

    DEFINE_FWK_MODULE( GenParticleSelector );
  }
}
