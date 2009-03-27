#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Python/src/PythonFilter.h"

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Python/src/PythonService.h"

DEFINE_FWK_SERVICE(PythonService);
DEFINE_FWK_MODULE(PythonFilter);
