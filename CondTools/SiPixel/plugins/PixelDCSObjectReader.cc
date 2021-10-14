#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondTools/SiPixel/plugins/PixelDCSObjectReader.h"
#include "CondFormats/DataRecord/interface/PixelDCSRcds.h"
#include "CondFormats/SiPixelObjects/interface/PixelDCSObject.h"

DEFINE_FWK_MODULE(PixelDCSObjectReader<PixelCaenChannelRcd>);
