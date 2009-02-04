//#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "SiStripElectronProducer.h"
//#include "SiStripElectronAssociator.h"
#include "FastSimulation/EgammaElectronAlgos/plugins/FastElectronSeedProducer.h"
//#include "PixelMatchElectronProducer.h"
//#include "PixelMatchGsfElectronProducer.h"
//#include "TrackProducerWithSeedAssoc.h"
//#include "CkfTrackCandidateMakerWithSeedAssoc.h"
//#include "CkfTrajectoryBuilderWithSeedAssocESProducer.h"

//using cms::CkfTrackCandidateMakerWithSeedAssoc;

DEFINE_SEAL_MODULE();

//DEFINE_ANOTHER_FWK_EVENTSETUP_MODULE(CkfTrajectoryBuilderWithSeedAssocESProducer);
//DEFINE_ANOTHER_FWK_MODULE(SiStripElectronProducer);
//DEFINE_ANOTHER_FWK_MODULE(SiStripElectronAssociator);
DEFINE_ANOTHER_FWK_MODULE(FastElectronSeedProducer);
//DEFINE_ANOTHER_FWK_MODULE(PixelMatchElectronProducer);
//DEFINE_ANOTHER_FWK_MODULE(PixelMatchGsfElectronProducer);
//DEFINE_ANOTHER_FWK_MODULE(TrackProducerWithSeedAssoc);
//DEFINE_ANOTHER_FWK_MODULE(GsfTrackProducerWithSeedAssoc);
