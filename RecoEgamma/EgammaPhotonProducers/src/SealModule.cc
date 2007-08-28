#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonProducer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonCorrectionProducer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/ConvertedPhotonProducer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionTrackCandidateProducer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/TrackProducerWithSCAssociation.h"




DEFINE_SEAL_MODULE();


DEFINE_ANOTHER_FWK_MODULE(PhotonProducer);
DEFINE_ANOTHER_FWK_MODULE(PhotonCorrectionProducer);
DEFINE_ANOTHER_FWK_MODULE(ConvertedPhotonProducer);
DEFINE_ANOTHER_FWK_MODULE(ConversionTrackCandidateProducer);
DEFINE_ANOTHER_FWK_MODULE(TrackProducerWithSCAssociation);
