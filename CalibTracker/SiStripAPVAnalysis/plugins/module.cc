#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_SEAL_MODULE();

#include "CalibTracker/SiStripAPVAnalysis/interface/ApvFactoryService.h"

DEFINE_ANOTHER_FWK_SERVICE(ApvFactoryService);
  
