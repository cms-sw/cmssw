#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
#include "EgammaTrackExtractor.h"
#include "EgammaEcalExtractor.h"
#include "EgammaHcalExtractor.h"
#include "EgammaTowerExtractor.h"
#include "EgammaRecHitExtractor.h"
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, egammaisolation::EgammaTrackExtractor,  "EgammaTrackExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, egammaisolation::EgammaEcalExtractor,   "EgammaEcalExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, egammaisolation::EgammaHcalExtractor,   "EgammaHcalExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, egammaisolation::EgammaTowerExtractor,  "EgammaTowerExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, egammaisolation::EgammaRecHitExtractor, "EgammaRecHitExtractor");

#include "EgammaEcalRecHitIsolationProducer.h"
#include "EgammaElectronTkIsolationProducer.h"
#include "EgammaElectronTkNumIsolationProducer.h"
#include "EgammaPhotonTkIsolationProducer.h"
#include "EgammaPhotonTkNumIsolationProducer.h"
#include "EgammaTowerIsolationProducer.h"
//#include "EgammaDetIdCollectionProducer.h"
#include "GamIsoDetIdCollectionProducer.h"
#include "EleIsoDetIdCollectionProducer.h"
#include "ParticleBasedIsoProducer.h"
#include "EgammaIsoHcalDetIdCollectionProducer.h"
#include "EgammaIsoESDetIdCollectionProducer.h"

DEFINE_FWK_MODULE(ParticleBasedIsoProducer);
DEFINE_FWK_MODULE(EgammaElectronTkIsolationProducer);
DEFINE_FWK_MODULE(EgammaElectronTkNumIsolationProducer);
DEFINE_FWK_MODULE(EgammaPhotonTkIsolationProducer);
DEFINE_FWK_MODULE(EgammaPhotonTkNumIsolationProducer);
DEFINE_FWK_MODULE(EgammaTowerIsolationProducer);
DEFINE_FWK_MODULE(EgammaEcalRecHitIsolationProducer);
//DEFINE_FWK_MODULE(EgammaDetIdCollectionProducer);
DEFINE_FWK_MODULE(EleIsoDetIdCollectionProducer);
DEFINE_FWK_MODULE(GamIsoDetIdCollectionProducer);
DEFINE_FWK_MODULE(EgammaIsoHcalDetIdCollectionProducer);
DEFINE_FWK_MODULE(EgammaIsoESDetIdCollectionProducer);

