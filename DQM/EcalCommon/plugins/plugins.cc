
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQM/EcalCommon/interface/EcalMonitorPrescaler.h"

DEFINE_FWK_MODULE(EcalMonitorPrescaler);

#include "DQM/EcalCommon/interface/EcalDQMStatusWriter.h"

DEFINE_FWK_MODULE(EcalDQMStatusWriter);

#include "DQM/EcalCommon/interface/EcalDQMStatusReader.h"

DEFINE_FWK_MODULE(EcalDQMStatusReader);

