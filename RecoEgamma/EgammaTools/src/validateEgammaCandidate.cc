#include "RecoEgamma/EgammaTools/interface/validateEgammaCandidate.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

void egammaTools::validateGsfElectron(reco::GsfElectron const& electron) {
  if (electron.convVtxFitProb() > 1.0f) {
    throw cms::Exception("EgammaError")
        << "invalid value " << electron.convVtxFitProb() << " for reco::GsfElectron.conversionRejection_.vtxFitProb\n"
        << "It should be either beween 0.0f and 1.0f or negative in case no matching conversion was found.\n"
        << "Probably you need to update the electron collection with the EG9X105XObjectUpdateModifier plugin:\n"
        << "see PhysicsTools/NanoAOD/python/electrons_cff.py for an example.\n";
  }
}
