#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

// The module providing event information 
#include "DQMServices/Components/src/DQMEventInfo.h"
DEFINE_ANOTHER_FWK_MODULE(DQMEventInfo);
#include "DQMServices/Components/interface/QualityTester.h"
DEFINE_ANOTHER_FWK_MODULE(QualityTester);
#include "DQMServices/Components/src/DQMFileSaver.h"
DEFINE_ANOTHER_FWK_MODULE(DQMFileSaver);
#include "DQMServices/Components/src/DQMFEDIntegrityClient.h"
DEFINE_ANOTHER_FWK_MODULE(DQMFEDIntegrityClient);
#include "DQMServices/Components/src/DQMStoreStats.h"
DEFINE_ANOTHER_FWK_MODULE(DQMStoreStats);
#include "DQMServices/Components/src/DQMLogError.h"
DEFINE_ANOTHER_FWK_MODULE(DQMLogError);

// Data Certification module for DAQ info
#include "DQMServices/Components/src/DQMDaqInfo.h"
DEFINE_ANOTHER_FWK_MODULE(DQMDaqInfo);

// module converting between ME and ROOT in Run tree of edm file
#include "DQMServices/Components/plugins/MEtoEDMConverter.h"
DEFINE_ANOTHER_FWK_MODULE(MEtoEDMConverter);
#include "DQMServices/Components/plugins/EDMtoMEConverter.h"
DEFINE_ANOTHER_FWK_MODULE(EDMtoMEConverter);

