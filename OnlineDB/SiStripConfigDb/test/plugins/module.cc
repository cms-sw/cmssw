// Last commit: $Id: $

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "OnlineDB/SiStripConfigDb/test/plugins/test_DatabaseService.h"
DEFINE_ANOTHER_FWK_MODULE(test_DatabaseService);

#include "OnlineDB/SiStripConfigDb/test/plugins/PopulateConfigDb.h"
DEFINE_ANOTHER_FWK_MODULE(PopulateConfigDb);
