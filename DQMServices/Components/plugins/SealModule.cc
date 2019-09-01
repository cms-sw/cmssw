#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// The module providing event information
#include "DQMEventInfo.h"
DEFINE_FWK_MODULE(DQMEventInfo);
#include "DQMServices/Components/interface/QualityTester.h"
DEFINE_FWK_MODULE(QualityTester);
#include "DQMFileSaver.h"
DEFINE_FWK_MODULE(DQMFileSaver);
#include "DQMFEDIntegrityClient.h"
DEFINE_FWK_MODULE(DQMFEDIntegrityClient);
#include "DQMStoreStats.h"
DEFINE_FWK_MODULE(DQMStoreStats);
#include "DQMMessageLogger.h"
DEFINE_FWK_MODULE(DQMMessageLogger);
#include "DQMMessageLoggerClient.h"
DEFINE_FWK_MODULE(DQMMessageLoggerClient);
#include "DQMFileReader.h"
DEFINE_FWK_MODULE(DQMFileReader);
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

#include "DQMServices/Components/plugins/MEtoMEComparitor.h"
//define this as a plug-in
DEFINE_FWK_MODULE(MEtoMEComparitor);
