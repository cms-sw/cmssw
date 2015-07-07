#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondTools/Geometry/plugins/XMLGeometryBuilder.h"
DEFINE_FWK_MODULE(XMLGeometryBuilder);

#include "CondTools/Geometry/plugins/PGeometricDetBuilder.h"
DEFINE_FWK_MODULE(PGeometricDetBuilder);

#include "CondTools/Geometry/plugins/PGeometricDetExtraBuilder.h"
DEFINE_FWK_MODULE(PGeometricDetExtraBuilder);

#include "CondTools/Geometry/plugins/PCaloGeometryBuilder.h"
DEFINE_FWK_MODULE(PCaloGeometryBuilder);

#include "CondTools/Geometry/plugins/CSCRecoIdealDBLoader.h"
DEFINE_FWK_MODULE(CSCRecoIdealDBLoader);

#include "CondTools/Geometry/plugins/DTRecoIdealDBLoader.h"
DEFINE_FWK_MODULE(DTRecoIdealDBLoader);

#include "CondTools/Geometry/plugins/RPCRecoIdealDBLoader.h"
DEFINE_FWK_MODULE(RPCRecoIdealDBLoader);

#include "CondTools/Geometry/plugins/PTrackerParametersDBBuilder.h"
DEFINE_FWK_MODULE(PTrackerParametersDBBuilder);

#include "CondTools/Geometry/plugins/HcalParametersDBBuilder.h"
DEFINE_FWK_MODULE(HcalParametersDBBuilder);
