#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "CondCore/PopCon/interface/PopConAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

//ReWritten DCS O2O
#include "CalibTracker/SiStripDCS/interface/SiStripDetVOffHandler.h"
typedef popcon::PopConAnalyzer<popcon::SiStripDetVOffHandler> SiStripPopConDetVOff;
DEFINE_ANOTHER_FWK_MODULE(SiStripPopConDetVOff);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripDCS/interface/SiStripDetVOffBuilder.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripDetVOffBuilder);


