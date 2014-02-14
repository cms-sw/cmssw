#include "Geometry/HGCalCommonData/plugins/DDHGCalEEAlgo.h"
#include "Geometry/HGCalCommonData/plugins/DDShashlikModule.h"
#include "Geometry/HGCalCommonData/plugins/DDShashlikSupermodule.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalEEAlgo, "hgcal:DDHGCalEEAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDShashlikModule, "shashlik:DDShashlikModule");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDShashlikSupermodule, "shashlik:DDShashlikSupermodule");
