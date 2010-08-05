#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <DQM/RPCMonitorClient/interface/RPCEventSummary.h>
DEFINE_FWK_MODULE(RPCEventSummary);

#include <DQM/RPCMonitorClient/interface/RPCMon_SS_Dbx_Global.h>
DEFINE_FWK_MODULE(RPCMon_SS_Dbx_Global);

#include <DQM/RPCMonitorClient/interface/RPCFEDIntegrity.h>
DEFINE_FWK_MODULE(RPCFEDIntegrity);

#include <DQM/RPCMonitorClient/interface/RPCMonitorRaw.h>
DEFINE_FWK_MODULE(RPCMonitorRaw);

#include <DQM/RPCMonitorClient/interface/RPCMonitorLinkSynchro.h>
DEFINE_FWK_MODULE(RPCMonitorLinkSynchro);

#include <DQM/RPCMonitorClient/interface/RPCDaqInfo.h>
DEFINE_FWK_MODULE(RPCDaqInfo);

#include <DQM/RPCMonitorClient/interface/RPCDcsInfoClient.h>
DEFINE_FWK_MODULE(RPCDcsInfoClient);

#include <DQM/RPCMonitorClient/interface/RPCDCSSummary.h>
DEFINE_FWK_MODULE(RPCDCSSummary);

#include <DQM/RPCMonitorClient/interface/RPCDataCertification.h>
DEFINE_FWK_MODULE(RPCDataCertification);

#include <DQM/RPCMonitorClient/interface/RPCDqmClient.h>
DEFINE_FWK_MODULE(RPCDqmClient);

#include <DQM/RPCMonitorClient/interface/RPCChamberQuality.h>
DEFINE_FWK_MODULE(RPCChamberQuality);


#include <DQM/RPCMonitorClient/interface/RPCEfficiencySecond.h>
DEFINE_FWK_MODULE(RPCEfficiencySecond);


