#include "FWCore/Framework/interface/MakerMacros.h"
// The module providing event information 
#include "DQMServices/Components/interface/DQMOnlineEnvironment.h"
DEFINE_FWK_MODULE(DQMOnlineEnvironment);
#include "DQMServices/Components/interface/QualityTester.h"
DEFINE_ANOTHER_FWK_MODULE(QualityTester);
#include "DQMServices/Components/src/DQMFileSaver.h"
DEFINE_ANOTHER_FWK_MODULE(DQMFileSaver);



