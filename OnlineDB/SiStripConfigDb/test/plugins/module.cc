// Last commit: $Id: module.cc,v 1.1 2007/05/11 12:05:08 bainbrid Exp $

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "OnlineDB/SiStripConfigDb/test/plugins/testDatabaseService.h"
DEFINE_ANOTHER_FWK_MODULE(testDatabaseService);

#include "OnlineDB/SiStripConfigDb/test/plugins/PopulateConfigDb.h"
DEFINE_ANOTHER_FWK_MODULE(PopulateConfigDb);

