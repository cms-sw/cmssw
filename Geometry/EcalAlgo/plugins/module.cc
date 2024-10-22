#include "Geometry/CaloEventSetup/interface/CaloGeometryEP.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

template class CaloGeometryEP<EcalBarrelGeometry, DDCompactView>;
template class CaloGeometryEP<EcalEndcapGeometry, DDCompactView>;
template class CaloGeometryEP<EcalPreshowerGeometry, DDCompactView>;

typedef CaloGeometryEP<EcalBarrelGeometry, DDCompactView> EcalBarrelGeometryEP;
DEFINE_FWK_EVENTSETUP_MODULE(EcalBarrelGeometryEP);

typedef CaloGeometryEP<EcalEndcapGeometry, DDCompactView> EcalEndcapGeometryEP;
DEFINE_FWK_EVENTSETUP_MODULE(EcalEndcapGeometryEP);

typedef CaloGeometryEP<EcalPreshowerGeometry, DDCompactView> EcalPreshowerGeometryEP;
DEFINE_FWK_EVENTSETUP_MODULE(EcalPreshowerGeometryEP);

#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBReader.h"

template class CaloGeometryDBEP<EcalBarrelGeometry, CaloGeometryDBReader>;
template class CaloGeometryDBEP<EcalEndcapGeometry, CaloGeometryDBReader>;
template class CaloGeometryDBEP<EcalPreshowerGeometry, CaloGeometryDBReader>;

typedef CaloGeometryDBEP<EcalBarrelGeometry, CaloGeometryDBReader> EcalBarrelGeometryFromDBEP;

DEFINE_FWK_EVENTSETUP_MODULE(EcalBarrelGeometryFromDBEP);

typedef CaloGeometryDBEP<EcalEndcapGeometry, CaloGeometryDBReader> EcalEndcapGeometryFromDBEP;

DEFINE_FWK_EVENTSETUP_MODULE(EcalEndcapGeometryFromDBEP);

typedef CaloGeometryDBEP<EcalPreshowerGeometry, CaloGeometryDBReader> EcalPreshowerGeometryFromDBEP;

DEFINE_FWK_EVENTSETUP_MODULE(EcalPreshowerGeometryFromDBEP);
