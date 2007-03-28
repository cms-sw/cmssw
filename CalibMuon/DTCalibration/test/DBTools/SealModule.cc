


#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DumpDBToFile.h"
#include "DumpFileToDB.h"
//#include "DTT0Analyzer.h"
#include "ProduceFakeDB.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(DumpDBToFile);
DEFINE_ANOTHER_FWK_MODULE(DumpFileToDB);
//DEFINE_ANOTHER_FWK_MODULE(DTT0Analyzer);
DEFINE_ANOTHER_FWK_MODULE(ProduceFakeDB);


