
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1Trigger/TextToDigi/src/TextToRaw.h"
#include "L1Trigger/TextToDigi/src/RctDigiToSourceCardText.h"
#include "L1Trigger/TextToDigi/src/SourceCardTextToRctDigi.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TextToRaw);
DEFINE_ANOTHER_FWK_MODULE(RctDigiToSourceCardText);
DEFINE_ANOTHER_FWK_MODULE(SourceCardTextToRctDigi);

