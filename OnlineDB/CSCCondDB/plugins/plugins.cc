#include "FWCore/Framework/interface/SourceFactory.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberIndexValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCCrateMapValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCChamberTimeCorrectionsValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCDDUMapValues.h"

DEFINE_FWK_EVENTSETUP_SOURCE(CSCChamberMapValues);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCChamberIndexValues);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCCrateMapValues);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCDDUMapValues);
DEFINE_FWK_EVENTSETUP_SOURCE(CSCChamberTimeCorrectionsValues);
