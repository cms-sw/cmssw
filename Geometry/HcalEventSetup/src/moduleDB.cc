#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"

#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBReader.h"

template class CaloGeometryDBEP< HcalGeometry , CaloGeometryDBReader> ;

typedef CaloGeometryDBEP< HcalGeometry , CaloGeometryDBReader> 
HcalGeometryFromDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(HcalGeometryFromDBEP);

template class CaloGeometryDBEP< CaloTowerGeometry , CaloGeometryDBReader> ;

typedef CaloGeometryDBEP< CaloTowerGeometry , CaloGeometryDBReader> 
CaloTowerGeometryFromDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(CaloTowerGeometryFromDBEP);
