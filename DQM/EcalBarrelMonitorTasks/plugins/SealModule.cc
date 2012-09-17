#include "FWCore/Framework/interface/MakerMacros.h"

#include "../interface/EcalDQMonitorTask.h"

DEFINE_FWK_MODULE(EcalDQMonitorTask);

#include "../interface/EcalFEDMonitor.h"

DEFINE_FWK_MODULE(EcalFEDMonitor);

#include "../interface/EBHltTask.h"

DEFINE_FWK_MODULE(EBHltTask);
