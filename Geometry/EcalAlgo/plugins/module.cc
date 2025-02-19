#include "Geometry/CaloEventSetup/interface/CaloGeometryEP.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"

template class CaloGeometryEP< EcalBarrelGeometry    > ;
template class CaloGeometryEP< EcalEndcapGeometry    > ;
template class CaloGeometryEP< EcalPreshowerGeometry > ;

typedef CaloGeometryEP< EcalBarrelGeometry > EcalBarrelGeometryEP ;
DEFINE_FWK_EVENTSETUP_MODULE(EcalBarrelGeometryEP);

typedef CaloGeometryEP< EcalEndcapGeometry > EcalEndcapGeometryEP ;
DEFINE_FWK_EVENTSETUP_MODULE(EcalEndcapGeometryEP);

typedef CaloGeometryEP< EcalPreshowerGeometry > EcalPreshowerGeometryEP ;
DEFINE_FWK_EVENTSETUP_MODULE(EcalPreshowerGeometryEP);

#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBReader.h"


template class CaloGeometryDBEP< EcalBarrelGeometry    , CaloGeometryDBReader> ;
template class CaloGeometryDBEP< EcalEndcapGeometry    , CaloGeometryDBReader> ;
template class CaloGeometryDBEP< EcalPreshowerGeometry , CaloGeometryDBReader> ;

typedef CaloGeometryDBEP< EcalBarrelGeometry , CaloGeometryDBReader> 
EcalBarrelGeometryFromDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(EcalBarrelGeometryFromDBEP);

typedef CaloGeometryDBEP< EcalEndcapGeometry , CaloGeometryDBReader> 
EcalEndcapGeometryFromDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(EcalEndcapGeometryFromDBEP);

typedef CaloGeometryDBEP< EcalPreshowerGeometry , CaloGeometryDBReader> 
EcalPreshowerGeometryFromDBEP ;

DEFINE_FWK_EVENTSETUP_MODULE(EcalPreshowerGeometryFromDBEP);
