#include "FWCore/Framework/interface/MakerMacros.h"
// The module providing event information 
#include "DQMServices/Components/src/DQMEventInfo.h"
DEFINE_FWK_MODULE(DQMEventInfo);
#include "DQMServices/Components/interface/QualityTester.h"
DEFINE_ANOTHER_FWK_MODULE(QualityTester);
#include "DQMServices/Components/src/DQMFileSaver.h"
DEFINE_ANOTHER_FWK_MODULE(DQMFileSaver);



