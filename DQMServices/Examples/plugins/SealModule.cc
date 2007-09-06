#include "FWCore/Framework/interface/InputSourceMacros.h"
// The example source module
#include "DQMServices/Examples/interface/DQMSourceExample.h"
// The example client module for running the client in the same application as the source
#include "DQMServices/Examples/interface/MonitorClient.h"
DEFINE_FWK_MODULE(MonitorClient);
DEFINE_ANOTHER_FWK_MODULE(DQMSourceExample);


