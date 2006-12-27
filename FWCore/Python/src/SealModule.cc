#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Python/src/PythonFilter.h"

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Python/src/PythonService.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE(PythonService);
DEFINE_ANOTHER_FWK_MODULE(PythonFilter);
