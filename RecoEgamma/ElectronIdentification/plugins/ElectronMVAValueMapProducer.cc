#include "RecoEgamma/EgammaTools/interface/MVAValueMapProducer.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

typedef MVAValueMapProducer<reco::GsfElectron> ElectronMVAValueMapProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronMVAValueMapProducer);
