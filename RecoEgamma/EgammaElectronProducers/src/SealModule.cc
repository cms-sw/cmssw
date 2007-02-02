#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "RecoEgamma/EgammaElectronProducers/interface/SiStripElectronProducer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/SiStripElectronAssociator.h"
#include "RecoEgamma/EgammaElectronProducers/interface/SiStripElectronAnalyzer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronProducer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronAnalyzer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronPixelSeedProducer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/ElectronPixelSeedAnalyzer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/PixelMatchElectronProducer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/PixelMatchElectronAnalyzer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/PixelMatchGsfElectronProducer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/PixelMatchGsfElectronAnalyzer.h"
#include "RecoEgamma/EgammaElectronProducers/interface/TrackProducerWithSeedAssoc.h"
#include "RecoEgamma/EgammaElectronProducers/interface/GsfTrackProducerWithSeedAssoc.h"
#include "RecoEgamma/EgammaElectronProducers/interface/CkfTrackCandidateMakerWithSeedAssoc.h"
#include "RecoEgamma/EgammaElectronProducers/interface/CkfTrajectoryBuilderWithSeedAssocESProducer.h"

using cms::CkfTrackCandidateMakerWithSeedAssoc;

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(CkfTrajectoryBuilderWithSeedAssocESProducer);
DEFINE_ANOTHER_FWK_MODULE(ElectronProducer);
DEFINE_ANOTHER_FWK_MODULE(SiStripElectronProducer);
DEFINE_ANOTHER_FWK_MODULE(SiStripElectronAssociator);
DEFINE_ANOTHER_FWK_MODULE(SiStripElectronAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(ElectronAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(ElectronPixelSeedProducer);
DEFINE_ANOTHER_FWK_MODULE(PixelMatchElectronProducer);
DEFINE_ANOTHER_FWK_MODULE(PixelMatchElectronAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(PixelMatchGsfElectronProducer);
DEFINE_ANOTHER_FWK_MODULE(PixelMatchGsfElectronAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(ElectronPixelSeedAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(TrackProducerWithSeedAssoc);
DEFINE_ANOTHER_FWK_MODULE(GsfTrackProducerWithSeedAssoc);
DEFINE_ANOTHER_FWK_MODULE(CkfTrackCandidateMakerWithSeedAssoc);
