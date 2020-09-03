#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CalibTracker/SiStripCommon/plugins/SiStripDetInfoFileWriter.h"

DEFINE_FWK_MODULE(SiStripDetInfoFileWriter);

#include "ShallowTree.h"
#include "ShallowEventDataProducer.h"
#include "ShallowDigisProducer.h"
#include "ShallowTrackClustersProducer.h"
#include "ShallowRechitClustersProducer.h"
#include "ShallowSimhitClustersProducer.h"
#include "ShallowTracksProducer.h"
#include "ShallowGainCalibration.h"
#include "ShallowSimTracksProducer.h"

DEFINE_FWK_MODULE(ShallowTree);
DEFINE_FWK_MODULE(ShallowEventDataProducer);
DEFINE_FWK_MODULE(ShallowDigisProducer);
DEFINE_FWK_MODULE(ShallowTrackClustersProducer);
DEFINE_FWK_MODULE(ShallowRechitClustersProducer);
DEFINE_FWK_MODULE(ShallowSimhitClustersProducer);
DEFINE_FWK_MODULE(ShallowTracksProducer);
DEFINE_FWK_MODULE(ShallowSimTracksProducer);
DEFINE_FWK_MODULE(ShallowGainCalibration);
