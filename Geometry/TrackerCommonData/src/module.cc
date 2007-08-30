//<<<<<< INCLUDES                                                       >>>>>>

#include "Geometry/TrackerCommonData/interface/DDPixBarLayerAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDPixFwdBlades.h"
#include "Geometry/TrackerCommonData/interface/DDTECAxialCableAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTECCoolAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTECModuleAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTECOptoHybAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTECPhiAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTECPhiAltAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTIBLayerAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTIBRadCableAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTIDAxialCableAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTIDModuleAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTIDModulePosAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTIDRingAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTOBRodAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTOBRadCableAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTOBAxCableAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerAngular.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerLinear.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerLinearXY.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerPhiAltAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerPhiAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerZPosAlgo.h"
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
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTIBRadCableAlgo,   "track:DDTIBRadCableAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTIDAxialCableAlgo, "track:DDTIDAxialCableAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTIDModuleAlgo,     "track:DDTIDModuleAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTIDModulePosAlgo,  "track:DDTIDModulePosAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTIDRingAlgo,       "track:DDTIDRingAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTOBRodAlgo,        "track:DDTOBRodAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTOBRadCableAlgo,   "track:DDTOBRadCableAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTOBAxCableAlgo,    "track:DDTOBAxCableAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerAngular,    "track:DDTrackerAngular");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerLinear,     "track:DDTrackerLinear");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerLinearXY,   "track:DDTrackerLinearXY");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerPhiAltAlgo, "track:DDTrackerPhiAltAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerPhiAlgo,    "track:DDTrackerPhiAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDTrackerZPosAlgo,   "track:DDTrackerZPosAlgo");
