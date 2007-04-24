#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_SEAL_MODULE();

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
DEFINE_ANOTHER_FWK_SERVICE(SiStripConfigDb);
