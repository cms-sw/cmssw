//<<<<<< INCLUDES                                                       >>>>>>

#include "Geometry/TrackerCommonData/plugins/DDPixBarLayerAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDPixFwdBlades.h"
#include "Geometry/TrackerCommonData/plugins/DDTECAxialCableAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTECCoolAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTECModuleAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTECOptoHybAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTECPhiAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTECPhiAltAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTIBLayerAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTIDAxialCableAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTIDModuleAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTIDModulePosAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTIDRingAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTOBAxCableAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTOBRodAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTOBRadCableAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTrackerAngularV1.h"
#include "Geometry/TrackerCommonData/plugins/DDTrackerAngular.h"
#include "Geometry/TrackerCommonData/plugins/DDTrackerLinear.h"
#include "Geometry/TrackerCommonData/plugins/DDTrackerLinearXY.h"
#include "Geometry/TrackerCommonData/plugins/DDTrackerPhiAltAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTrackerPhiAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTrackerZPosAlgo.h"
#include "Geometry/TrackerCommonData/plugins/DDTrackerXYZPosAlgo.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDPixBarLayerAlgo,   "track:DDPixBarLayerAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDPixFwdBlades,      "track:DDPixFwdBlades");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTECAxialCableAlgo, "track:DDTECAxialCableAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTECCoolAlgo,       "track:DDTECCoolAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTECModuleAlgo,     "track:DDTECModuleAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTECOptoHybAlgo,    "track:DDTECOptoHybAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTECPhiAlgo,        "track:DDTECPhiAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTECPhiAltAlgo,     "track:DDTECPhiAltAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTIBLayerAlgo,      "track:DDTIBLayerAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTIDAxialCableAlgo, "track:DDTIDAxialCableAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTIDModuleAlgo,     "track:DDTIDModuleAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTIDModulePosAlgo,  "track:DDTIDModulePosAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTIDRingAlgo,       "track:DDTIDRingAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTOBAxCableAlgo,    "track:DDTOBAxCableAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTOBRodAlgo,        "track:DDTOBRodAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTOBRadCableAlgo,   "track:DDTOBRadCableAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerAngular,    "track:DDTrackerAngular");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerAngularV1,  "track:DDTrackerAngularV1");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerLinear,     "track:DDTrackerLinear");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerLinearXY,   "track:DDTrackerLinearXY");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerPhiAltAlgo, "track:DDTrackerPhiAltAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerPhiAlgo,    "track:DDTrackerPhiAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerZPosAlgo,   "track:DDTrackerZPosAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerXYZPosAlgo, "track:DDTrackerXYZPosAlgo");
