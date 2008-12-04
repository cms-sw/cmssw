#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_SEAL_MODULE();

#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripFedCablingBuilderFromDb);

#include "OnlineDB/SiStripESSources/interface/SiStripPedestalsBuilderFromDb.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripPedestalsBuilderFromDb);

#include "OnlineDB/SiStripESSources/interface/SiStripNoiseBuilderFromDb.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripNoiseBuilderFromDb);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripCondObjBuilderFromDb);
