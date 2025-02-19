#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberIndexValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCCrateMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberTimeCorrectionsValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCDDUMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCMap1Read.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberTimeCorrectionsReadTest.h"


DEFINE_FWK_MODULE(CSCMap1Read);
DEFINE_FWK_MODULE(CSCChamberTimeCorrectionsReadTest);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCChamberMapValues);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCChamberIndexValues);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCCrateMapValues);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCDDUMapValues);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCChamberTimeCorrectionsValues);
