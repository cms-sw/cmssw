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

// module converting between ME and ROOT in Run tree of edm file
#include "DQMServices/Components/plugins/MEtoROOTConverter.h"
DEFINE_ANOTHER_FWK_MODULE(MEtoROOTConverter);
#include "DQMServices/Components/plugins/ROOTtoMEConverter.h"
DEFINE_ANOTHER_FWK_MODULE(ROOTtoMEConverter);

