#define DEBUG 0
#define COUT if (DEBUG) cout
//<<<<<< INCLUDES                                                       >>>>>>

#include "Geometry/TrackerCommonData/interface/DDPixBarLayerAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTECAxialCableAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTECCoolAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTECModuleAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTECOptoHybAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTECPhiAltAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTIBLayerAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTIBRadCableAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTIDAxialCableAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTIDModuleAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTIDRingAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTOBRodAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTOBRadCableAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerAngular.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerLinear.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerLinearXY.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerPhiAltAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerPhiAlgo.h"
#include "Geometry/TrackerCommonData/interface/DDTrackerZPosAlgo.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "PluginManager/ModuleDef.h"

DEFINE_SEAL_MODULE ();
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDPixBarLayerAlgo, "track:DDPixBarLayerAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTECAxialCableAlgo, "track:DDTECAxialCableAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTECCoolAlgo, "track:DDTECCoolAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTECModuleAlgo, "track:DDTECModuleAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTECOptoHybAlgo, "track:DDTECOptoHybAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTECPhiAltAlgo, "track:DDTECPhiAltAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTIBLayerAlgo, "track:DDTIBLayerAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTIBRadCableAlgo, "track:DDTIBRadCableAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTIDAxialCableAlgo, "track:DDTIDAxialCableAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTIDModuleAlgo, "track:DDTIDModuleAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTIDRingAlgo, "track:DDTIDRingAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTOBRodAlgo, "track:DDTOBRodAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTOBRadCableAlgo, "track:DDTOBRadCableAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTrackerAngular, "track:DDTrackerAngular");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTrackerLinear, "track:DDTrackerLinear");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTrackerLinearXY, "track:DDTrackerLinearXY");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTrackerPhiAltAlgo, "track:DDTrackerPhiAltAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTrackerPhiAlgo, "track:DDTrackerPhiAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDTrackerZPosAlgo, "track:DDTrackerZPosAlgo");
