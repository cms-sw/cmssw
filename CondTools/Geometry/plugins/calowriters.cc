#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBWriter.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"

#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

 
template class CaloGeometryDBEP< EcalBarrelGeometry    , CaloGeometryDBWriter> ;
template class CaloGeometryDBEP< EcalEndcapGeometry    , CaloGeometryDBWriter> ;
template class CaloGeometryDBEP< EcalPreshowerGeometry , CaloGeometryDBWriter> ;

template class CaloGeometryDBEP< HcalGeometry          , CaloGeometryDBWriter> ;

typedef CaloGeometryDBEP< EcalBarrelGeometry , CaloGeometryDBWriter> 
EcalBarrelGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(EcalBarrelGeometryToDBEP);

typedef CaloGeometryDBEP< EcalEndcapGeometry , CaloGeometryDBWriter> 
EcalEndcapGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(EcalEndcapGeometryToDBEP);

typedef CaloGeometryDBEP< EcalPreshowerGeometry , CaloGeometryDBWriter> 
EcalPreshowerGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(EcalPreshowerGeometryToDBEP);

typedef CaloGeometryDBEP< HcalGeometry , CaloGeometryDBWriter> 
HcalGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(HcalGeometryToDBEP);
