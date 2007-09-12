#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <EventFilter/CSCTFRawToDigi/interface/CSCTFUnpacker.h>
#include <EventFilter/CSCTFRawToDigi/interface/CSCTFPacker.h>
#include <EventFilter/CSCTFRawToDigi/interface/CSCTFAnalyzer.h>

DEFINE_FWK_MODULE(CSCTFUnpacker);
DEFINE_ANOTHER_FWK_MODULE(CSCTFPacker);
DEFINE_ANOTHER_FWK_MODULE(CSCTFAnalyzer);

