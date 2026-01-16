#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWIO/RNTupleTempOutput/interface/RNTupleTempOutputModule.h"
#include "FWIO/RNTupleTempOutput/interface/TimeoutRNTupleTempOutputModule.h"

using edm::rntuple_temp::RNTupleTempOutputModule;
using edm::rntuple_temp::TimeoutRNTupleTempOutputModule;
DEFINE_FWK_MODULE(RNTupleTempOutputModule);
DEFINE_FWK_MODULE(TimeoutRNTupleTempOutputModule);
