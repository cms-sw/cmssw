#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondTools/SiStrip/plugins/SiStripThresholdBuilder.h"
DEFINE_FWK_MODULE(SiStripThresholdBuilder);

#include "CondTools/SiStrip/plugins/SiStripThresholdReader.h"
DEFINE_FWK_MODULE(SiStripThresholdReader);

#include "CondTools/SiStrip/plugins/SiStripPedestalsBuilder.h"
DEFINE_FWK_MODULE(SiStripPedestalsBuilder);

#include "CondTools/SiStrip/plugins/SiStripPedestalsReader.h"
DEFINE_FWK_MODULE(SiStripPedestalsReader);

#include "CondTools/SiStrip/plugins/SiStripNoisesBuilder.h"
DEFINE_FWK_MODULE(SiStripNoisesBuilder);

#include "CondTools/SiStrip/plugins/SiStripNoisesReader.h"
DEFINE_FWK_MODULE(SiStripNoisesReader);

#include "CondTools/SiStrip/plugins/SiStripApvGainBuilder.h"
DEFINE_FWK_MODULE(SiStripApvGainBuilder);

#include "CondTools/SiStrip/plugins/SiStripApvGainReader.h"
DEFINE_FWK_MODULE(SiStripApvGainReader);

#include "CondTools/SiStrip/plugins/SiStripBadChannelBuilder.h"
DEFINE_FWK_MODULE(SiStripBadChannelBuilder);

#include "CondTools/SiStrip/plugins/SiStripBadFiberBuilder.h"
DEFINE_FWK_MODULE(SiStripBadFiberBuilder);

#include "CondTools/SiStrip/plugins/SiStripDetVOffFakeBuilder.h"
DEFINE_FWK_MODULE(SiStripDetVOffFakeBuilder);

#include "CondTools/SiStrip/plugins/SiStripDetVOffReader.h"
DEFINE_FWK_MODULE(SiStripDetVOffReader);

#include "CondTools/SiStrip/plugins/SiStripCablingTrackerMap.h"
DEFINE_FWK_MODULE(SiStripCablingTrackerMap);

#include "CondTools/SiStrip/plugins/SiStripFedCablingBuilder.h"
DEFINE_FWK_MODULE(SiStripFedCablingBuilder);

#include "CondTools/SiStrip/plugins/SiStripFedCablingReader.h"
DEFINE_FWK_MODULE(SiStripFedCablingReader);

#include "CondTools/SiStrip/plugins/SiStripLorentzAngleReader.h"
DEFINE_FWK_MODULE(SiStripLorentzAngleReader);

#include "CondTools/SiStrip/plugins/SiStripSummaryReader.h"
DEFINE_FWK_MODULE(SiStripSummaryReader);

#include "CondTools/SiStrip/plugins/SiStripSummaryBuilder.h"
DEFINE_FWK_MODULE(SiStripSummaryBuilder);
