#include "FWCore/Framework/interface/MakerMacros.h"
#include <EventFilter/CSCRawToDigi/interface/CSCDCCUnpacker.h>
#include <EventFilter/CSCRawToDigi/interface/DigiAnalyzer.h>
#include <EventFilter/CSCRawToDigi/src/CSCDigiToRawModule.h>
#include <EventFilter/CSCRawToDigi/src/CSCDigiToPattern.h>
#include <EventFilter/CSCRawToDigi/interface/CSCViewDigi.h>

DEFINE_FWK_MODULE(CSCDCCUnpacker);
DEFINE_ANOTHER_FWK_MODULE(DigiAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(CSCDigiToRawModule);
DEFINE_ANOTHER_FWK_MODULE(CSCDigiToPattern);
DEFINE_ANOTHER_FWK_MODULE(CSCViewDigi);
