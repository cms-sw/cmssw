#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningSource.h"
DEFINE_FWK_MODULE(SiStripCommissioningSource);

#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningRunTypeFilter.h"
DEFINE_FWK_MODULE(SiStripCommissioningRunTypeFilter);

#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningSeedFilter.h"
DEFINE_FWK_MODULE(SiStripCommissioningSeedFilter);

#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningBasicPrescaler.h"
DEFINE_FWK_MODULE(SiStripCommissioningBasicPrescaler);
