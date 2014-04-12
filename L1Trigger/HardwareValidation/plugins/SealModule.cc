#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1Trigger/HardwareValidation/interface/L1Comparator.h"
DEFINE_FWK_MODULE(L1Comparator);

#include <L1Trigger/HardwareValidation/plugins/L1DEFilter.h>
DEFINE_FWK_MODULE(L1DEFilter);

#include <L1Trigger/HardwareValidation/plugins/L1DummyProducer.h>
DEFINE_FWK_MODULE(L1DummyProducer);

#include <L1Trigger/HardwareValidation/plugins/L1EmulBias.h>
DEFINE_FWK_MODULE(L1EmulBias);
