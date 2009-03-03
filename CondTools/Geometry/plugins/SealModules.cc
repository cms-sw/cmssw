#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();
#include "CondTools/Geometry/plugins/XMLGeometryBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(XMLGeometryBuilder);

#include "CondTools/Geometry/plugins/PGeometricDetBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(PGeometricDetBuilder);

#include "CondTools/Geometry/plugins/PCaloGeometryBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(PCaloGeometryBuilder);

#include "CondTools/Geometry/plugins/CSCRecoIdealDBLoader.h"
DEFINE_ANOTHER_FWK_MODULE(CSCRecoIdealDBLoader);

#include "CondTools/Geometry/plugins/DTRecoIdealDBLoader.h"
DEFINE_ANOTHER_FWK_MODULE(DTRecoIdealDBLoader);

#include "CondTools/Geometry/plugins/RPCRecoIdealDBLoader.h"
DEFINE_ANOTHER_FWK_MODULE(RPCRecoIdealDBLoader);
