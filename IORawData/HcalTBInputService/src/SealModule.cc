#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "IORawData/HcalTBInputService/interface/HcalTBSource.h"
#include "IORawData/HcalTBInputService/src/HcalTBWriter.h"


DEFINE_FWK_INPUT_SOURCE(HcalTBSource);
DEFINE_FWK_MODULE(HcalTBWriter);
