#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningSource.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripCommissioningSource);

#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningRunTypeFilter.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripCommissioningRunTypeFilter);

#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningSeedFilter.h"
DEFINE_ANOTHER_FWK_MODULE(SiStripCommissioningSeedFilter);

