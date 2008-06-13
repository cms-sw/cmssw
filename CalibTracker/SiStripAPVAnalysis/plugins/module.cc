#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripAPVAnalysis/interface/ApvFactoryService.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/PedFactoryService.h"

DEFINE_ANOTHER_FWK_SERVICE(ApvFactoryService);
DEFINE_ANOTHER_FWK_SERVICE(PedFactoryService);
  
