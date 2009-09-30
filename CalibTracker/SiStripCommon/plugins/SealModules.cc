#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/Framework/interface/ModuleFactory.h"
//#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

#include "CalibTracker/SiStripCommon/plugins/SiStripDetInfoFileWriter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(SiStripDetInfoFileWriter);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripDetInfoFileReader);

#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
DEFINE_ANOTHER_FWK_SERVICE(TkDetMap);


#include "CalibTracker/SiStripCommon/interface/ShallowTree.h"
#include "CalibTracker/SiStripCommon/interface/ShallowEventDataProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowDigisProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowClustersProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowTrackClustersProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowRechitClustersProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowSimhitClustersProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowTracksProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowSimTracksProducer.h"

DEFINE_ANOTHER_FWK_MODULE(ShallowTree);
DEFINE_ANOTHER_FWK_MODULE(ShallowEventDataProducer);
DEFINE_ANOTHER_FWK_MODULE(ShallowDigisProducer);
DEFINE_ANOTHER_FWK_MODULE(ShallowClustersProducer);
DEFINE_ANOTHER_FWK_MODULE(ShallowTrackClustersProducer);
DEFINE_ANOTHER_FWK_MODULE(ShallowRechitClustersProducer);
DEFINE_ANOTHER_FWK_MODULE(ShallowSimhitClustersProducer);
DEFINE_ANOTHER_FWK_MODULE(ShallowTracksProducer);
DEFINE_ANOTHER_FWK_MODULE(ShallowSimTracksProducer);
