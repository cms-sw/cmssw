#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVHandler.h"

typedef popcon::PopConAnalyzer<popcon::SiStripModuleHVHandler> SiStripPopConModuleHV;
DEFINE_ANOTHER_FWK_MODULE(SiStripPopConModuleHV);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripDCS/interface/SiStripModuleHVBuilder.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripModuleHVBuilder);

