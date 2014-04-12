#include "Geometry/EcalTestBeam/test/ee/CaloGeometryEPtest.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

template class CaloGeometryEPtest< EcalEndcapGeometry    > ;

typedef CaloGeometryEPtest< EcalEndcapGeometry > testEcalEndcapGeometryEP ;
DEFINE_FWK_EVENTSETUP_MODULE(testEcalEndcapGeometryEP);
