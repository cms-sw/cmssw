#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondCore/PopCon/interface/PopConAnalyzer.h"

#include "OnlineDB/SiStripO2O/plugins/SiStripPopConConfigDbObjHandler.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConConfigDbObjHandler<SiStripFedCabling> > SiStripPopConFedCabling;
DEFINE_FWK_MODULE(SiStripPopConFedCabling);

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConConfigDbObjHandler<SiStripNoises> > SiStripPopConNoise;
DEFINE_FWK_MODULE(SiStripPopConNoise);

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConConfigDbObjHandler<SiStripPedestals> > SiStripPopConPedestals;
DEFINE_FWK_MODULE(SiStripPopConPedestals);

#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConConfigDbObjHandler<SiStripThreshold> > SiStripPopConThreshold;
DEFINE_FWK_MODULE(SiStripPopConThreshold);

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConConfigDbObjHandler<SiStripBadStrip> > SiStripPopConBadStrip;
DEFINE_FWK_MODULE(SiStripPopConBadStrip);

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConConfigDbObjHandler<SiStripApvGain> > SiStripPopConApvGain;
DEFINE_FWK_MODULE(SiStripPopConApvGain);

#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConConfigDbObjHandler<SiStripLatency> > SiStripPopConApvLatency;
DEFINE_FWK_MODULE(SiStripPopConApvLatency);

#include "OnlineDB/SiStripO2O/plugins/SiStripPopConHandlerUnitTestNoise.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConHandlerUnitTestNoise<SiStripNoises> > SiStripPopConNoiseUnitTest;
DEFINE_FWK_MODULE(SiStripPopConNoiseUnitTest);

#include "OnlineDB/SiStripO2O/plugins/SiStripPopConHandlerUnitTestGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
typedef popcon::PopConAnalyzer<popcon::SiStripPopConHandlerUnitTestGain<SiStripApvGain> > SiStripPopConApvGainUnitTest;
DEFINE_FWK_MODULE(SiStripPopConApvGainUnitTest);
