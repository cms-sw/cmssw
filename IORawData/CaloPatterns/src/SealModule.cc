#include "IORawData/CaloPatterns/src/HcalPatternSource.h"
#include "IORawData/CaloPatterns/src/HtrXmlPattern.h"

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_INPUT_SOURCE(HcalPatternSource);
DEFINE_ANOTHER_FWK_MODULE(HtrXmlPattern);
