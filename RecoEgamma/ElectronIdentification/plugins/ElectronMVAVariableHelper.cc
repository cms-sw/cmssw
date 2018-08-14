#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAVariableHelper.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "FWCore/Framework/interface/MakerMacros.h"

typedef ElectronMVAVariableHelper<reco::GsfElectron> GsfElectronMVAVariableHelper;

DEFINE_FWK_MODULE(GsfElectronMVAVariableHelper);
