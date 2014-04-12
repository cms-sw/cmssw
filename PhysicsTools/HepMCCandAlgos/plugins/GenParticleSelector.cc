/* \class GenParticleSelector
 * 
 * Configurable GenParticle Selector
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

typedef SingleObjectSelector<
          reco::GenParticleCollection,
          StringCutObjectSelector<reco::GenParticle>
        > GenParticleSelector;

DEFINE_FWK_MODULE(GenParticleSelector);

