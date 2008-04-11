// Last commit: $Id: module.cc,v 1.3 2008/03/26 09:13:11 bainbrid Exp $

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "OnlineDB/SiStripConfigDb/test/plugins/testSiStripConfigDb.h"
DEFINE_ANOTHER_FWK_MODULE(testSiStripConfigDb);

