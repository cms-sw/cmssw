#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <EventFilter/CSCTFRawToDigi/interface/CSCTFUnpacker.h>
#include <EventFilter/CSCTFRawToDigi/interface/CSCTFPacker.h>
#include <EventFilter/CSCTFRawToDigi/interface/CSCTFValidator.h>

DEFINE_FWK_MODULE(CSCTFUnpacker);
DEFINE_FWK_MODULE(CSCTFPacker);
DEFINE_FWK_MODULE(CSCTFValidator);

