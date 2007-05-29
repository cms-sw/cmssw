#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
DEFINE_SEAL_MODULE();

#include "IORawData/SiStripInputSources/interface/TBRUInputSource.h"
DEFINE_ANOTHER_FWK_INPUT_SOURCE(TBRUInputSource);

