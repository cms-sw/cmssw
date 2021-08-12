#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncTOFCorr.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncT0Only.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFromDB.h"

DEFINE_EDM_PLUGIN(DTTTrigSyncFactory, DTTTrigSyncTOFCorr, "DTTTrigSyncTOFCorr");
DEFINE_EDM_PLUGIN(DTTTrigSyncFactory, DTTTrigSyncT0Only, "DTTTrigSyncT0Only");
DEFINE_EDM_PLUGIN(DTTTrigSyncFactory, DTTTrigSyncFromDB, "DTTTrigSyncFromDB");
