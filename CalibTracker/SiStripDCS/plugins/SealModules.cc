#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"



#include "CondCore/PopCon/interface/PopConAnalyzer.h"

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

//ReWritten DCS O2O
#include "CalibTracker/SiStripDCS/interface/SiStripDetVOffHandler.h"
typedef popcon::PopConAnalyzer<popcon::SiStripDetVOffHandler> SiStripPopConDetVOff;
DEFINE_FWK_MODULE(SiStripPopConDetVOff);

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "CalibTracker/SiStripDCS/interface/SiStripDetVOffBuilder.h"
DEFINE_FWK_SERVICE(SiStripDetVOffBuilder);


