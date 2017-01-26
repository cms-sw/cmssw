#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonCoreProducer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonProducer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/ConvertedPhotonProducer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionTrackCandidateProducer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/TrackProducerWithSCAssociation.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionProducer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionTrackProducer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionTrackMerger.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/GEDPhotonCoreProducer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/GEDPhotonProducer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/ReducedEGProducer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/ConversionGSCrysFixer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/GEDPhotonCoreGSCrysFixer.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/GEDPhotonGSCrysFixer.h"

DEFINE_FWK_MODULE(PhotonCoreProducer);
DEFINE_FWK_MODULE(PhotonProducer);
DEFINE_FWK_MODULE(ConvertedPhotonProducer);
DEFINE_FWK_MODULE(ConversionTrackCandidateProducer);
DEFINE_FWK_MODULE(TrackProducerWithSCAssociation);
DEFINE_FWK_MODULE(ConversionProducer);
DEFINE_FWK_MODULE(ConversionTrackProducer);
DEFINE_FWK_MODULE(ConversionTrackMerger);
DEFINE_FWK_MODULE(GEDPhotonCoreProducer);
DEFINE_FWK_MODULE(GEDPhotonProducer);
DEFINE_FWK_MODULE(ReducedEGProducer);
DEFINE_FWK_MODULE(ConversionGSCrysFixer);
DEFINE_FWK_MODULE(GEDPhotonCoreGSCrysFixer);
DEFINE_FWK_MODULE(GEDPhotonGSCrysFixer);
