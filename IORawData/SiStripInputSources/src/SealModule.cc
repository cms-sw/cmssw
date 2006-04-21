#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IORawData/SiStripInputSources/interface/TBRUInputSource.h"
#include "IORawData/SiStripInputSources/interface/TBMonitorInputSource.h"
#include "IORawData/SiStripInputSources/interface/CommissioningInputSource.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(TBRUInputSource)
DEFINE_ANOTHER_FWK_INPUT_SOURCE(TBMonitorInputSource)
DEFINE_ANOTHER_FWK_INPUT_SOURCE(CommissioningInputSource)
