#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_SEAL_MODULE();

#include "CalibMuon/RPCCalibration/interface/RPCFakeCalibration.h"
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(RPCFakeCalibration);
