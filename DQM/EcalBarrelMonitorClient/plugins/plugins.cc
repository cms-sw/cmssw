
#include "FWCore/Framework/interface/MakerMacros.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunDat.h"
#endif

#include "DQM/EcalBarrelMonitorClient/interface/EcalBarrelMonitorClient.h"

DEFINE_FWK_MODULE(EcalBarrelMonitorClient);

#include "DQM/EcalBarrelMonitorClient/interface/EBTrendClient.h"

DEFINE_FWK_MODULE(EBTrendClient);

