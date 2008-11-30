#include "CondFormats/SiPixelObjects/interface/PixelDCSObject.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

EVENTSETUP_DATA_REG(PixelDCSObject<bool>);
EVENTSETUP_DATA_REG(PixelDCSObject<float>);
EVENTSETUP_DATA_REG(PixelDCSObject<CaenChannel>);
