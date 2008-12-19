#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();
#include "CondTools/Geometry/plugins/PGeometricDetBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(PGeometricDetBuilder);

#include "CondTools/Geometry/plugins/PEcalGeometryBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(PEcalGeometryBuilder);

#include "CondTools/Geometry/plugins/XMLGeometryBuilder.h"
DEFINE_ANOTHER_FWK_MODULE(XMLGeometryBuilder);

#include "CondTools/Geometry/plugins/CSCRecoIdealDBLoader.h"
DEFINE_ANOTHER_FWK_MODULE(CSCRecoIdealDBLoader);

#include "CondTools/Geometry/plugins/DTRecoIdealDBLoader.h"
DEFINE_FWK_MODULE(DTRecoIdealDBLoader);
