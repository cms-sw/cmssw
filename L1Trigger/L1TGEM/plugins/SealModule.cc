#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "L1Trigger/L1TGEM/plugins/GEMPadDigiProducer.h"
#include "L1Trigger/L1TGEM/plugins/ME0PadDigiProducer.h"
#include "L1Trigger/L1TGEM/plugins/GEMPadDigiClusterProducer.h"
#include "L1Trigger/L1TGEM/plugins/ME0PadDigiClusterProducer.h"
#include "L1Trigger/L1TGEM/plugins/ME0TriggerProducer.h"
#include "L1Trigger/L1TGEM/plugins/ME0TriggerPseudoProducer.h"

DEFINE_FWK_MODULE(GEMPadDigiProducer);
DEFINE_FWK_MODULE(ME0PadDigiProducer);
DEFINE_FWK_MODULE(GEMPadDigiClusterProducer);
DEFINE_FWK_MODULE(ME0PadDigiClusterProducer);
DEFINE_FWK_MODULE(ME0TriggerProducer);
DEFINE_FWK_MODULE(ME0TriggerPseudoProducer);
