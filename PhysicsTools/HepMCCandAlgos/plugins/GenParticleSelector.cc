/* \class GenParticleSelector
 * 
 * Configurable GenParticle Selector
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

typedef SingleObjectSelector<
          reco::GenParticleCollection,
          StringCutObjectSelector<reco::GenParticle>
        > GenParticleSelector;

DEFINE_FWK_MODULE(GenParticleSelector);
