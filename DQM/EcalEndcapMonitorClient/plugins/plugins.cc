
#include "FWCore/Framework/interface/MakerMacros.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunDat.h"
#endif

#include <DQM/EcalEndcapMonitorClient/interface/EcalEndcapMonitorClient.h>
#include <DQM/EcalEndcapMonitorClient/interface/EETrendClient.h>

DEFINE_FWK_MODULE(EcalEndcapMonitorClient);
DEFINE_FWK_MODULE(EETrendClient);

