/** \class reco::modules::GenParticleCustomSelector
 *
 *  Filter to select GenParticles
 *
 *  \author Giuseppe Cerati, UCSD
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/CandAlgos/interface/GenParticleCustomSelector.h"

namespace reco {
  namespace modules {
    typedef SingleObjectSelector<GenParticleCollection,::GenParticleCustomSelector> 
    GenParticleCustomSelector ;

    DEFINE_FWK_MODULE( GenParticleCustomSelector );
  }
}
