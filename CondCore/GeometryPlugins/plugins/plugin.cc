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
#include "Geometry/Records/interface/PHGCalRcd.h"
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
#include "Geometry/Records/interface/ME0RecoGeometryRcd.h"

#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"
#include "Geometry/Records/interface/PTrackerParametersRcd.h"

#include "CondFormats/GeometryObjects/interface/PTrackerAdditionalParametersPerDet.h"
#include "Geometry/Records/interface/PTrackerAdditionalParametersPerDetRcd.h"

#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"

#include "CondFormats/GeometryObjects/interface/PHGCalParameters.h"
#include "Geometry/Records/interface/PHGCalParametersRcd.h"

#include "CondFormats/GeometryObjects/interface/PDetGeomDesc.h"
#include "Geometry/Records/interface/VeryForwardIdealGeometryRecord.h"

REGISTER_PLUGIN(GeometryFileRcd, FileBlob);
REGISTER_PLUGIN(IdealGeometryRecord, PGeometricDet);
REGISTER_PLUGIN(PTrackerParametersRcd, PTrackerParameters);
REGISTER_PLUGIN(PTrackerAdditionalParametersPerDetRcd, PTrackerAdditionalParametersPerDet);
REGISTER_PLUGIN(PEcalBarrelRcd, PCaloGeometry);
REGISTER_PLUGIN_NO_SERIAL(PEcalEndcapRcd, PCaloGeometry);
REGISTER_PLUGIN_NO_SERIAL(PEcalPreshowerRcd, PCaloGeometry);
REGISTER_PLUGIN_NO_SERIAL(PHcalRcd, PCaloGeometry);
REGISTER_PLUGIN_NO_SERIAL(PHGCalRcd, PCaloGeometry);
REGISTER_PLUGIN(PHGCalParametersRcd, PHGCalParameters);
REGISTER_PLUGIN(HcalParametersRcd, HcalParameters);
REGISTER_PLUGIN_NO_SERIAL(PCaloTowerRcd, PCaloGeometry);
REGISTER_PLUGIN_NO_SERIAL(PZdcRcd, PCaloGeometry);
REGISTER_PLUGIN_NO_SERIAL(PCastorRcd, PCaloGeometry);
REGISTER_PLUGIN(CSCRecoDigiParametersRcd, CSCRecoDigiParameters);
REGISTER_PLUGIN(VeryForwardIdealGeometryRecord, PDetGeomDesc);
REGISTER_PLUGIN(CSCRecoGeometryRcd, RecoIdealGeometry);
REGISTER_PLUGIN_NO_SERIAL(DTRecoGeometryRcd, RecoIdealGeometry);
REGISTER_PLUGIN_NO_SERIAL(RPCRecoGeometryRcd, RecoIdealGeometry);
REGISTER_PLUGIN_NO_SERIAL(GEMRecoGeometryRcd, RecoIdealGeometry);
REGISTER_PLUGIN_NO_SERIAL(ME0RecoGeometryRcd, RecoIdealGeometry);
