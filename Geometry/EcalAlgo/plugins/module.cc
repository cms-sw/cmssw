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

#include "Geometry/CaloEventSetup/interface/CaloGeometryFromDBEP.h"


template class CaloGeometryFromDBEP< EcalBarrelGeometry    , true> ;
template class CaloGeometryFromDBEP< EcalEndcapGeometry    , true> ;
template class CaloGeometryFromDBEP< EcalPreshowerGeometry , true> ;

typedef CaloGeometryFromDBEP< EcalBarrelGeometry , false> EcalBarrelGeometryFromDBEP ;
DEFINE_FWK_EVENTSETUP_MODULE(EcalBarrelGeometryFromDBEP);

typedef CaloGeometryFromDBEP< EcalEndcapGeometry , false> EcalEndcapGeometryFromDBEP ;
DEFINE_FWK_EVENTSETUP_MODULE(EcalEndcapGeometryFromDBEP);

typedef CaloGeometryFromDBEP< EcalPreshowerGeometry , false> EcalPreshowerGeometryFromDBEP ;
DEFINE_FWK_EVENTSETUP_MODULE(EcalPreshowerGeometryFromDBEP);
