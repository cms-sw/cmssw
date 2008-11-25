#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
DEFINE_SEAL_MODULE();

#include "IORawData/SiStripInputSources/plugins/FEDRawDataAnalyzer.h"
DEFINE_ANOTHER_FWK_MODULE(FEDRawDataAnalyzer);
