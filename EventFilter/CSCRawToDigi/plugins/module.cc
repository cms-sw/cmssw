#include "FWCore/Framework/interface/MakerMacros.h"
#include <EventFilter/CSCRawToDigi/interface/CSCDCCUnpacker.h>
#include <EventFilter/CSCRawToDigi/interface/DigiAnalyzer.h>
#include <EventFilter/CSCRawToDigi/src/CSCDigiToRawModule.h>
#include <EventFilter/CSCRawToDigi/src/CSCDigiToPattern.h>
#include <EventFilter/CSCRawToDigi/interface/CSCViewDigi.h>
#include <EventFilter/CSCRawToDigi/interface/CSCDigiValidator.h>

DEFINE_FWK_MODULE(CSCDCCUnpacker);
DEFINE_FWK_MODULE(DigiAnalyzer);
DEFINE_FWK_MODULE(CSCDigiToRawModule);
DEFINE_FWK_MODULE(CSCDigiToPattern);
DEFINE_FWK_MODULE(CSCViewDigi);
DEFINE_FWK_MODULE(CSCDigiValidator);
