#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/Common/interface/FileBlob.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"

#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h" 

#include "CondFormats/GeometryObjects/interface/PCaloGeometry.h"
#include "Geometry/Records/interface/PEcalBarrelRcd.h"
#include "Geometry/Records/interface/PEcalEndcapRcd.h"
#include "Geometry/Records/interface/PEcalPreshowerRcd.h"
#include "Geometry/Records/interface/PHcalRcd.h"
#include "Geometry/Records/interface/PCaloTowerRcd.h"
#include "Geometry/Records/interface/PZdcRcd.h"
#include "Geometry/Records/interface/PCastorRcd.h"

#include "CondFormats/GeometryObjects/interface/CSCRecoDigiParameters.h"
#include "Geometry/Records/interface/CSCRecoDigiParametersRcd.h"

#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"    
#include "Geometry/Records/interface/CSCRecoGeometryRcd.h"
#include "Geometry/Records/interface/DTRecoGeometryRcd.h"
#include "Geometry/Records/interface/RPCRecoGeometryRcd.h"
#include "Geometry/Records/interface/GEMRecoGeometryRcd.h"

#include "CondFormats/GeometryObjects/interface/PGeometricDetExtra.h"
#include "Geometry/Records/interface/PGeometricDetExtraRcd.h"

#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"

#include "CondFormats/GeometryObjects/interface/PHcalParameters.h"
#include "Geometry/Records/interface/PHcalParametersRcd.h"

#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"

REGISTER_PLUGIN(GeometryFileRcd,FileBlob);
REGISTER_PLUGIN(IdealGeometryRecord,PGeometricDet);
REGISTER_PLUGIN(PGeometricDetExtraRcd,PGeometricDetExtra);
REGISTER_PLUGIN(PTrackerParametersRcd,PTrackerParameters);
REGISTER_PLUGIN(PEcalBarrelRcd,PCaloGeometry);
REGISTER_PLUGIN(PEcalEndcapRcd,PCaloGeometry);
REGISTER_PLUGIN(PEcalPreshowerRcd,PCaloGeometry);
REGISTER_PLUGIN(PHcalRcd,PCaloGeometry);
REGISTER_PLUGIN(PHcalParametersRcd,PHcalParameters);
REGISTER_PLUGIN(HcalParametersRcd,HcalParameters);
REGISTER_PLUGIN(PCaloTowerRcd,PCaloGeometry);
REGISTER_PLUGIN(PZdcRcd,PCaloGeometry);
REGISTER_PLUGIN(PCastorRcd,PCaloGeometry);
REGISTER_PLUGIN(CSCRecoDigiParametersRcd,CSCRecoDigiParameters);
REGISTER_PLUGIN(CSCRecoGeometryRcd,RecoIdealGeometry);
REGISTER_PLUGIN(DTRecoGeometryRcd,RecoIdealGeometry);
REGISTER_PLUGIN(RPCRecoGeometryRcd,RecoIdealGeometry);
REGISTER_PLUGIN(GEMRecoGeometryRcd,RecoIdealGeometry);
