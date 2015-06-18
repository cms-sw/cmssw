#include "RecoEgamma/EgammaTools/interface/MVAValueMapProducer.h"

typedef MVAValueMapProducer<reco::GsfElectron> ElectronMVAValueMapProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronMVAValueMapProducer);
