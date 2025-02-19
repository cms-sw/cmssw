#include "CommonTools/CandAlgos/interface/GenJetParticleSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
namespace reco {
  namespace modules {

    typedef SingleObjectSelector<
              reco::GenParticleCollection,
              ::GenJetParticleSelector,
              reco::GenParticleRefVector
            > GenJetParticleRefSelector;

   DEFINE_FWK_MODULE(GenJetParticleRefSelector);

  }
}
