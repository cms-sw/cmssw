#include "Geometry/CaloEventSetup/interface/CaloGeometryEP.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

template class CaloGeometryEP<EcalBarrelGeometry, cms::DDCompactView>;
template class CaloGeometryEP<EcalEndcapGeometry, cms::DDCompactView>;
template class CaloGeometryEP<EcalPreshowerGeometry, cms::DDCompactView>;

typedef CaloGeometryEP<EcalBarrelGeometry, cms::DDCompactView> EcalBarrelGeometryEPdd4hep;
DEFINE_FWK_EVENTSETUP_MODULE(EcalBarrelGeometryEPdd4hep);

typedef CaloGeometryEP<EcalEndcapGeometry, cms::DDCompactView> EcalEndcapGeometryEPdd4hep;
DEFINE_FWK_EVENTSETUP_MODULE(EcalEndcapGeometryEPdd4hep);

typedef CaloGeometryEP<EcalPreshowerGeometry, cms::DDCompactView> EcalPreshowerGeometryEPdd4hep;
DEFINE_FWK_EVENTSETUP_MODULE(EcalPreshowerGeometryEPdd4hep);
