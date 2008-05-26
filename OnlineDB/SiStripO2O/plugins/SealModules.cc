#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "CondCore/PopCon/interface/PopConAnalyzer.h"

#include "OnlineDB/SiStripO2O/plugins/SiStripPopConNoiseHandler.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConNoiseHandler> SiStripPopConNoiseTest;
DEFINE_ANOTHER_FWK_MODULE(SiStripPopConNoiseTest);


#include "OnlineDB/SiStripO2O/plugins/SiStripPopConConfigDbObjHandler.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConConfigDbObjHandler<SiStripFedCabling> > SiStripPopConFedCabling;
DEFINE_ANOTHER_FWK_MODULE(SiStripPopConFedCabling);

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConConfigDbObjHandler<SiStripNoises> > SiStripPopConNoise;
DEFINE_ANOTHER_FWK_MODULE(SiStripPopConNoise);

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConConfigDbObjHandler<SiStripPedestals> > SiStripPopConPedestals;
DEFINE_ANOTHER_FWK_MODULE(SiStripPopConPedestals);

#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConConfigDbObjHandler<SiStripThreshold> > SiStripPopConThreshold;
DEFINE_ANOTHER_FWK_MODULE(SiStripPopConThreshold);

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConConfigDbObjHandler<SiStripBadStrip> > SiStripPopConBadStrip;
DEFINE_ANOTHER_FWK_MODULE(SiStripPopConBadStrip);






