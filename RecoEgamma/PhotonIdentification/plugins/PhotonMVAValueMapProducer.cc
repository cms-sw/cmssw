#include "RecoEgamma/EgammaTools/interface/MVAValueMapProducer.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"

typedef MVAValueMapProducer<reco::Photon> PhotonMVAValueMapProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(PhotonMVAValueMapProducer);
