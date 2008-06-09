#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "DQM/SiStripCommissioningSources/plugins/tracking/SiStripFineDelayHit.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripFineDelayHit);

#include "DQM/SiStripCommissioningSources/plugins/tracking/ClusterCount.h"
DEFINE_ANOTHER_FWK_MODULE(ClusterCount);

