// Last commit: $Id: module.cc,v 1.2 2007/11/21 13:45:48 bainbrid Exp $

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "OnlineDB/SiStripConfigDb/test/plugins/testSiStripConfigDb.h"
DEFINE_ANOTHER_FWK_MODULE(testSiStripConfigDb);

#include "OnlineDB/SiStripConfigDb/test/plugins/PopulateConfigDb.h"
DEFINE_ANOTHER_FWK_MODULE(PopulateConfigDb);

