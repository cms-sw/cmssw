#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBWriter.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"

 
template class CaloGeometryDBEP< EcalBarrelGeometry    , CaloGeometryDBWriter> ;
template class CaloGeometryDBEP< EcalEndcapGeometry    , CaloGeometryDBWriter> ;
template class CaloGeometryDBEP< EcalPreshowerGeometry , CaloGeometryDBWriter> ;

template class CaloGeometryDBEP< HcalGeometry          , CaloGeometryDBWriter> ;
template class CaloGeometryDBEP< CaloTowerGeometry     , CaloGeometryDBWriter> ;
template class CaloGeometryDBEP< ZdcGeometry           , CaloGeometryDBWriter> ;
template class CaloGeometryDBEP< CastorGeometry        , CaloGeometryDBWriter> ;

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

typedef CaloGeometryDBEP< CaloTowerGeometry , CaloGeometryDBWriter> 
CaloTowerGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(CaloTowerGeometryToDBEP);

typedef CaloGeometryDBEP< ZdcGeometry , CaloGeometryDBWriter> 
ZdcGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(ZdcGeometryToDBEP);

typedef CaloGeometryDBEP< CastorGeometry , CaloGeometryDBWriter> 
CastorGeometryToDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(CastorGeometryToDBEP);
