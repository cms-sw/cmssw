#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"


#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripFedCablingBuilderFromDb);

#include "OnlineDB/SiStripESSources/interface/SiStripPedestalsBuilderFromDb.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripPedestalsBuilderFromDb);

#include "OnlineDB/SiStripESSources/interface/SiStripNoiseBuilderFromDb.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripNoiseBuilderFromDb);

#include "OnlineDB/SiStripESSources/interface/SiStripGainBuilderFromDb.h"
DEFINE_FWK_EVENTSETUP_SOURCE(SiStripGainBuilderFromDb);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"
DEFINE_FWK_SERVICE(SiStripCondObjBuilderFromDb);
