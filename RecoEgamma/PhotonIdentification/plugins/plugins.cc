#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "RecoEgamma/PhotonIdentification/plugins/PhotonIDProducer.h"
#include "PhysicsTools/SelectorUtils/interface/VersionedIdProducer.h"
#include "RecoEgamma/EgammaTools/interface/MVAValueMapProducer.h"

typedef VersionedIdProducer<reco::PhotonPtr> VersionedPhotonIdProducer;
typedef MVAValueMapProducer<reco::Photon> PhotonMVAValueMapProducer;

DEFINE_FWK_MODULE(VersionedPhotonIdProducer);
DEFINE_FWK_MODULE(PhotonMVAValueMapProducer);
DEFINE_FWK_MODULE(PhotonIDProducer);
