#include "FWCore/Framework/interface/MakerMacros.h"
// The example source module
#include "DQMServices/Examples/interface/DQMSourceExample.h"
// The example client module for running the client in the same application as 
// the source
#include "DQMServices/Examples/interface/DQMClientExample.h"
DEFINE_FWK_MODULE(DQMClientExample);
DEFINE_FWK_MODULE(DQMSourceExample);
#include <DQMServices/Examples/interface/ConverterTester.h>
DEFINE_FWK_MODULE(ConverterTester);
#include <DQMServices/Examples/interface/HarvestingAnalyzer.h>
DEFINE_FWK_MODULE(HarvestingAnalyzer);
#include <DQMServices/Examples/interface/HarvestingDataCertification.h>
DEFINE_FWK_MODULE(HarvestingDataCertification);

