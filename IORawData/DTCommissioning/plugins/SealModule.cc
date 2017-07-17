#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "IORawData/DTCommissioning/plugins/DTNewROS8FileReader.h"
#include "IORawData/DTCommissioning/plugins/DTROS8FileReader.h"
#include "IORawData/DTCommissioning/plugins/DTROS25FileReader.h"
#include "IORawData/DTCommissioning/plugins/DTDDUFileReader.h"
#include "IORawData/DTCommissioning/plugins/DTSpyReader.h"

DEFINE_FWK_MODULE ( DTNewROS8FileReader);
DEFINE_FWK_MODULE ( DTROS8FileReader);
DEFINE_FWK_MODULE ( DTROS25FileReader);
DEFINE_FWK_MODULE ( DTDDUFileReader);
DEFINE_FWK_MODULE ( DTSpyReader);
