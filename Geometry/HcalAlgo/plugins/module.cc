//<<<<<< INCLUDES                                                       >>>>>>

#include "Geometry/HcalAlgo/plugins/DDHCalAngular.h"
#include "Geometry/HcalAlgo/plugins/DDHCalBarrelAlgo.h"
#include "Geometry/HcalAlgo/plugins/DDHCalEndcapAlgo.h"
#include "Geometry/HcalAlgo/plugins/DDHCalEndcapModuleAlgo.h"
#include "Geometry/HcalAlgo/plugins/DDHCalFibreBundle.h"
#include "Geometry/HcalAlgo/plugins/DDHCalForwardAlgo.h"
#include "Geometry/HcalAlgo/plugins/DDHCalLinearXY.h"
#include "Geometry/HcalAlgo/plugins/DDHCalTBCableAlgo.h"
#include "Geometry/HcalAlgo/plugins/DDHCalTBZposAlgo.h"
#include "Geometry/HcalAlgo/plugins/DDHCalTestBeamAlgo.h"
#include "Geometry/HcalAlgo/plugins/DDHCalXtalAlgo.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalAngular,      "hcal:DDHCalAngular");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalBarrelAlgo,   "hcal:DDHCalBarrelAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalEndcapAlgo,   "hcal:DDHCalEndcapAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalEndcapModuleAlgo,"hcal:DDHCalEndcapModuleAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalForwardAlgo,  "hcal:DDHCalForwardAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalFibreBundle,  "hcal:DDHCalFibreBundle");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalLinearXY,     "hcal:DDHCalLinearXY");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalTBCableAlgo,  "hcal:DDHCalTBCableAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalTBZposAlgo,   "hcal:DDHCalTBZposAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalTestBeamAlgo, "hcal:DDHCalTestBeamAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHCalXtalAlgo,     "hcal:DDHCalXtalAlgo");
