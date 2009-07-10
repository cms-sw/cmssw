#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberIndexValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCCrateMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCDDUMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCMap1Read.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CSCMap1Read);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCChamberMapValues);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCChamberIndexValues);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCCrateMapValues);
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(CSCDDUMapValues);
