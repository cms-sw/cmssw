#include "FWCore/Framework/interface/MakerMacros.h"
// The example source module
#include "DQMServices/Examples/interface/DQMSourceExample.h"
// The example client module for running the client in the same application as 
// the source
#include "DQMServices/Examples/interface/DQMClientExample.h"
DEFINE_FWK_MODULE(DQMClientExample);
DEFINE_ANOTHER_FWK_MODULE(DQMSourceExample);
#include <DQMServices/Examples/interface/ConverterTester.h>
DEFINE_ANOTHER_FWK_MODULE(ConverterTester);
#include <DQMServices/Examples/interface/PostConverterAnalyzer.h>
DEFINE_ANOTHER_FWK_MODULE(PostConverterAnalyzer);
#include <DQMServices/Examples/interface/ConverterQualityTester.h>
DEFINE_ANOTHER_FWK_MODULE(ConverterQualityTester);
