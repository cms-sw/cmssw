#define DEBUG 0
#define COUT if (DEBUG) cout
//<<<<<< INCLUDES                                                       >>>>>>

#include "Geometry/TrackerSimData/interface/DDPixBarLayerAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTECAxialCableAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTECCoolAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTECModuleAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTECOptoHybAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTECPhiAltAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTIBLayerAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTIBRadCableAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTIDAxialCableAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTIDModuleAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTIDRingAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTOBRodAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTOBRadCableAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTrackerAngular.h"
#include "Geometry/TrackerSimData/interface/DDTrackerLinear.h"
#include "Geometry/TrackerSimData/interface/DDTrackerLinearXY.h"
#include "Geometry/TrackerSimData/interface/DDTrackerPhiAltAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTrackerPhiAlgo.h"
#include "Geometry/TrackerSimData/interface/DDTrackerZPosAlgo.h"
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
