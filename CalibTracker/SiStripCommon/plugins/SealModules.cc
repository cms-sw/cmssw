#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CalibTracker/SiStripCommon/plugins/SiStripDetInfoFileWriter.h"


DEFINE_FWK_MODULE(SiStripDetInfoFileWriter);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
DEFINE_FWK_SERVICE(SiStripDetInfoFileReader);


#include "CalibTracker/SiStripCommon/interface/ShallowTree.h"
#include "CalibTracker/SiStripCommon/interface/ShallowEventDataProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowDigisProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowClustersProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowTrackClustersProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowRechitClustersProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowSimhitClustersProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowTracksProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowGainCalibration.h"
#include "ShallowSimTracksProducer.h"

DEFINE_FWK_MODULE(ShallowTree);
DEFINE_FWK_MODULE(ShallowEventDataProducer);
DEFINE_FWK_MODULE(ShallowDigisProducer);
DEFINE_FWK_MODULE(ShallowClustersProducer);
DEFINE_FWK_MODULE(ShallowTrackClustersProducer);
DEFINE_FWK_MODULE(ShallowRechitClustersProducer);
DEFINE_FWK_MODULE(ShallowSimhitClustersProducer);
DEFINE_FWK_MODULE(ShallowTracksProducer);
DEFINE_FWK_MODULE(ShallowSimTracksProducer);
DEFINE_FWK_MODULE(ShallowGainCalibration);
