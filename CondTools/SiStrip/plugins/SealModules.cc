#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();
#include "CondTools/SiStrip/plugins/SiStripThresholdBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripThresholdBuilder);

#include "CondTools/SiStrip/plugins/SiStripThresholdReader.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripThresholdReader);


#include "CondTools/SiStrip/plugins/SiStripPedestalsBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripPedestalsBuilder);

#include "CondTools/SiStrip/plugins/SiStripPedestalsReader.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripPedestalsReader);

#include "CondTools/SiStrip/plugins/SiStripNoisesBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripNoisesBuilder);

#include "CondTools/SiStrip/plugins/SiStripNoisesReader.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripNoisesReader);

#include "CondTools/SiStrip/plugins/SiStripApvGainBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripApvGainBuilder);

#include "CondTools/SiStrip/plugins/SiStripApvGainReader.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripApvGainReader);

#include "CondTools/SiStrip/plugins/SiStripBadChannelBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripBadChannelBuilder);

#include "CondTools/SiStrip/plugins/SiStripBadFiberBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripBadFiberBuilder);

#include "CondTools/SiStrip/plugins/SiStripBadStripReader.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripBadStripReader);

#include "CondTools/SiStrip/plugins/SiStripPerformanceSummaryBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripPerformanceSummaryBuilder);

#include "CondTools/SiStrip/plugins/SiStripPerformanceSummaryReader.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripPerformanceSummaryReader);
