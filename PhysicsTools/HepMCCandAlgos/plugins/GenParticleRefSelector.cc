/* \class GenParticleRefSelector
 * 
 * GenParticle Selector based on a configurable cut.
 * saves a vector of references
 * Usage:
 * 
 * module selectedParticles = GenParticleRefSelector {
 *   InputTag src = genParticles
 *   string cut = "pt > 15.0"
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

typedef SingleObjectSelector<
          reco::GenParticleCollection,
          StringCutObjectSelector<reco::GenParticle>
       > GenParticleRefSelector;

DEFINE_FWK_MODULE(GenParticleRefSelector);

