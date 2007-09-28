//<<<<<< INCLUDES                                                       >>>>>>

#include "Geometry/HcalAlgo/interface/DDHCalAngular.h"
#include "Geometry/HcalAlgo/interface/DDHCalBarrelAlgo.h"
#include "Geometry/HcalAlgo/interface/DDHCalEndcapAlgo.h"
#include "Geometry/HcalAlgo/interface/DDHCalForwardAlgo.h"
#include "Geometry/HcalAlgo/interface/DDHCalLinearXY.h"
#include "Geometry/HcalAlgo/interface/DDHCalTBCableAlgo.h"
#include "Geometry/HcalAlgo/interface/DDHCalTBZposAlgo.h"
#include "Geometry/HcalAlgo/interface/DDHCalTestBeamAlgo.h"
#include "Geometry/HcalAlgo/interface/DDHCalXtalAlgo.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalAngular,      "hcal:DDHCalAngular");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalBarrelAlgo,   "hcal:DDHCalBarrelAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalEndcapAlgo,   "hcal:DDHCalEndcapAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalForwardAlgo,  "hcal:DDHCalForwardAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalLinearXY,     "hcal:DDHCalLinearXY");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalTBCableAlgo,  "hcal:DDHCalTBCableAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalTBZposAlgo,   "hcal:DDHCalTBZposAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalTestBeamAlgo, "hcal:DDHCalTestBeamAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalXtalAlgo,     "hcal:DDHCalXtalAlgo");
