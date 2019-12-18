#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMFEDIntegrityClient.h"
DEFINE_FWK_MODULE(DQMFEDIntegrityClient);
#include "DQMStoreStats.h"
DEFINE_FWK_MODULE(DQMStoreStats);
#include "DQMMessageLogger.h"
DEFINE_FWK_MODULE(DQMMessageLogger);
#include "DQMMessageLoggerClient.h"
DEFINE_FWK_MODULE(DQMMessageLoggerClient);
#include "DQMProvInfo.h"
DEFINE_FWK_MODULE(DQMProvInfo);
#include "DQMDcsInfo.h"
DEFINE_FWK_MODULE(DQMDcsInfo);
#include "DQMDcsInfoClient.h"
DEFINE_FWK_MODULE(DQMDcsInfoClient);
#include "DQMScalInfo.h"
DEFINE_FWK_MODULE(DQMScalInfo);

// Data Certification module for DAQ info
#include "DQMDaqInfo.h"
DEFINE_FWK_MODULE(DQMDaqInfo);

// module converting between ME and ROOT in Run tree of edm file
#include "DQMServices/Components/plugins/MEtoEDMConverter.h"
DEFINE_FWK_MODULE(MEtoEDMConverter);
#include "DQMServices/Components/plugins/EDMtoMEConverter.h"
DEFINE_FWK_MODULE(EDMtoMEConverter);
