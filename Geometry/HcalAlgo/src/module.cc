//<<<<<< INCLUDES                                                       >>>>>>

#include "Geometry/HcalAlgo/interface/DDHCalBarrelAlgo.h"
#include "Geometry/HcalAlgo/interface/DDHCalEndcapAlgo.h"
#include "Geometry/HcalAlgo/interface/DDHCalForwardAlgo.h"
#include "Geometry/HcalAlgo/interface/DDHCalTestBeamAlgo.h"
#include "Geometry/HcalAlgo/interface/DDHCalXtalAlgo.h"
#include "Geometry/HcalAlgo/interface/DDHCalTBCableAlgo.h"
#include "Geometry/HcalAlgo/interface/DDHCalAngular.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"
#include "PluginManager/ModuleDef.h"

DEFINE_SEAL_MODULE ();
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDHCalBarrelAlgo, "hcal:DDHCalBarrelAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDHCalEndcapAlgo, "hcal:DDHCalEndcapAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDHCalForwardAlgo, "hcal:DDHCalForwardAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDHCalTestBeamAlgo, "hcal:DDHCalTestBeamAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDHCalXtalAlgo, "hcal:DDHCalXtalAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDHCalTBCableAlgo, "hcal:DDHCalTBCableAlgo");
DEFINE_SEAL_PLUGIN (DDAlgorithmFactory, DDHCalAngular, "hcal:DDHCalAngular");
