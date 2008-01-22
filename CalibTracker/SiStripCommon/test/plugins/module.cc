#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripCommon/test/plugins/testSiStripFedIdListReader.h"
DEFINE_ANOTHER_FWK_MODULE(testSiStripFedIdListReader);
