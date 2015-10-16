#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"



// The module providing event information 
#include "DQMServices/Components/src/DQMEventInfo.h"
DEFINE_FWK_MODULE(DQMEventInfo);
#include "DQMServices/Components/interface/QualityTester.h"
DEFINE_FWK_MODULE(QualityTester);
#include "DQMServices/Components/src/DQMFileSaver.h"
DEFINE_FWK_MODULE(DQMFileSaver);
#include "DQMServices/Components/src/DQMFEDIntegrityClient.h"
DEFINE_FWK_MODULE(DQMFEDIntegrityClient);
#include "DQMServices/Components/src/DQMStoreStats.h"
DEFINE_FWK_MODULE(DQMStoreStats);
#include "DQMServices/Components/src/DQMMessageLogger.h"
DEFINE_FWK_MODULE(DQMMessageLogger);
#include "DQMServices/Components/src/DQMMessageLoggerClient.h"
DEFINE_FWK_MODULE(DQMMessageLoggerClient);
#include "DQMServices/Components/src/DQMFileReader.h"
DEFINE_FWK_MODULE(DQMFileReader);
#include "DQMServices/Components/src/DQMProvInfo.h"
DEFINE_FWK_MODULE(DQMProvInfo);
#include "DQMServices/Components/src/DQMDcsInfo.h"
DEFINE_FWK_MODULE(DQMDcsInfo);
#include "DQMServices/Components/src/DQMDcsInfoClient.h"
DEFINE_FWK_MODULE(DQMDcsInfoClient);
#include "DQMServices/Components/src/DQMScalInfo.h"
DEFINE_FWK_MODULE(DQMScalInfo);

// Data Certification module for DAQ info
#include "DQMServices/Components/src/DQMDaqInfo.h"
DEFINE_FWK_MODULE(DQMDaqInfo);

// module converting between ME and ROOT in Run tree of edm file
#include "DQMServices/Components/plugins/MEtoEDMConverter.h"
DEFINE_FWK_MODULE(MEtoEDMConverter);
#include "DQMServices/Components/plugins/EDMtoMEConverter.h"
DEFINE_FWK_MODULE(EDMtoMEConverter);

#include "DQMServices/Components/plugins/MEtoMEComparitor.h"
//define this as a plug-in
DEFINE_FWK_MODULE(MEtoMEComparitor);
