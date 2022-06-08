#include "FWCore/Framework/interface/MakerMacros.h"

//General Client
#include <DQM/RPCMonitorClient/interface/RPCDqmClient.h>
DEFINE_FWK_MODULE(RPCDqmClient);

#include <DQM/RPCMonitorClient/interface/RPCRecHitProbabilityClient.h>
DEFINE_FWK_MODULE(RPCRecHitProbabilityClient);

#include <DQM/RPCMonitorClient/interface/RPCDcsInfoClient.h>
DEFINE_FWK_MODULE(RPCDcsInfoClient);

#include <DQM/RPCMonitorClient/interface/RPCEventSummary.h>
DEFINE_FWK_MODULE(RPCEventSummary);

#include <DQM/RPCMonitorClient/interface/RPCDaqInfo.h>
DEFINE_FWK_MODULE(RPCDaqInfo);

#include <DQM/RPCMonitorClient/interface/RPCDCSSummary.h>
DEFINE_FWK_MODULE(RPCDCSSummary);

#include <DQM/RPCMonitorClient/interface/RPCDataCertification.h>
DEFINE_FWK_MODULE(RPCDataCertification);
