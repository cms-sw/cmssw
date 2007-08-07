#include "FWCore/Framework/interface/MakerMacros.h"

#include "TextToRaw.h"
#include "RawToText.h"
#include "RctDigiToSourceCardText.h"
#include "SourceCardTextToRctDigi.h"
#include "RctTextToRctDigi.h"
#include "RctDigiToRctText.h"

DEFINE_FWK_MODULE(TextToRaw);
DEFINE_FWK_MODULE(RawToText);
DEFINE_FWK_MODULE(RctDigiToSourceCardText);
DEFINE_FWK_MODULE(SourceCardTextToRctDigi);
DEFINE_FWK_MODULE(RctTextToRctDigi);
DEFINE_FWK_MODULE(RctDigiToRctText);

