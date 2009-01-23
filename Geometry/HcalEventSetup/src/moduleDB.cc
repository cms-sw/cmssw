#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBReader.h"

template class CaloGeometryDBEP< HcalGeometry , CaloGeometryDBReader> ;

typedef CaloGeometryDBEP< HcalGeometry , CaloGeometryDBReader> 
HcalGeometryFromDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(HcalGeometryFromDBEP);
