#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQM/CSCMonitorModule/interface/CSCMonitorModule.h"
#include "DQM/CSCMonitorModule/interface/CSCDaqInfo.h"
#include "DQM/CSCMonitorModule/interface/CSCDcsInfo.h"
#include "DQM/CSCMonitorModule/interface/CSCCertificationInfo.h"
#include "DQM/CSCMonitorModule/interface/CSCOfflineClient.h"

DEFINE_FWK_MODULE(CSCMonitorModule);
DEFINE_FWK_MODULE(CSCDaqInfo);
DEFINE_FWK_MODULE(CSCDcsInfo);
DEFINE_FWK_MODULE(CSCCertificationInfo);
DEFINE_FWK_MODULE(CSCOfflineClient);
