#include "FWCore/Framework/interface/MakerMacros.h"

#include "GctDigiToPsbText.h"
#include "GtPsbTextToDigi.h"
#include "RawToText.h"
#include "RctDigiToRctText.h"
#include "RctDigiToSourceCardText.h"
#include "RctTextToRctDigi.h"
#include "SourceCardTextToRctDigi.h"
#include "TextToRaw.h"

DEFINE_FWK_MODULE(TextToRaw);
DEFINE_FWK_MODULE(RawToText);
DEFINE_FWK_MODULE(RctDigiToSourceCardText);
DEFINE_FWK_MODULE(SourceCardTextToRctDigi);
DEFINE_FWK_MODULE(RctTextToRctDigi);
DEFINE_FWK_MODULE(RctDigiToRctText);
DEFINE_FWK_MODULE(GtPsbTextToDigi);
DEFINE_FWK_MODULE(GctDigiToPsbText);
