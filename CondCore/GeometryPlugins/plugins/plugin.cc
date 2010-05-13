#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/GeometryObjects/interface/GeometryFile.h"
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

#include "CondFormats/GeometryObjects/interface/PGeometricDetExtra.h"
#include "Geometry/Records/interface/GeometricDetExtraRcd.h"



REGISTER_PLUGIN(GeometryFileRcd,GeometryFile);
REGISTER_PLUGIN(IdealGeometryRecord,PGeometricDet);
REGISTER_PLUGIN(GeometricDetExtraRcd,PGeometricDetExtra);
REGISTER_PLUGIN(PEcalBarrelRcd,PCaloGeometry);
REGISTER_PLUGIN(PEcalEndcapRcd,PCaloGeometry);
REGISTER_PLUGIN(PEcalPreshowerRcd,PCaloGeometry);
REGISTER_PLUGIN(PHcalRcd,PCaloGeometry);
REGISTER_PLUGIN(PCaloTowerRcd,PCaloGeometry);
REGISTER_PLUGIN(PZdcRcd,PCaloGeometry);
REGISTER_PLUGIN(PCastorRcd,PCaloGeometry);
REGISTER_PLUGIN(CSCRecoDigiParametersRcd,CSCRecoDigiParameters);
REGISTER_PLUGIN(CSCRecoGeometryRcd,RecoIdealGeometry);
REGISTER_PLUGIN(DTRecoGeometryRcd,RecoIdealGeometry);
REGISTER_PLUGIN(RPCRecoGeometryRcd,RecoIdealGeometry);
