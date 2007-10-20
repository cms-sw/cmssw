
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <EventFilter/EcalTBRawToDigi/interface/EcalDCCUnpackingModule.h>
DEFINE_ANOTHER_FWK_MODULE(EcalDCCTBUnpackingModule);

#include <EventFilter/EcalTBRawToDigi/interface/EcalDCC07UnpackingModule.h>
DEFINE_ANOTHER_FWK_MODULE(EcalDCCTB07UnpackingModule);

