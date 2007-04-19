
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "TextToRaw.h"
#include "RctDigiToSourceCardText.h"
#include "SourceCardTextToRctDigi.h"
#include "RctTextToRctDigi.h"
#include "RctDigiToRctText.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TextToRaw);
DEFINE_ANOTHER_FWK_MODULE(RctDigiToSourceCardText);
DEFINE_ANOTHER_FWK_MODULE(SourceCardTextToRctDigi);
DEFINE_ANOTHER_FWK_MODULE(RctTextToRctDigi);
DEFINE_ANOTHER_FWK_MODULE(RctDigiToRctText);

