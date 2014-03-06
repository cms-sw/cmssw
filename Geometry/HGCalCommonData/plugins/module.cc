#include "Geometry/HGCalCommonData/plugins/DDHGCalEEAlgo.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalHEAlgo.h"
#include "Geometry/HGCalCommonData/plugins/DDShashlikModule.h"
#include "Geometry/HGCalCommonData/plugins/DDShashlikSupermodule.h"
#include "Geometry/HGCalCommonData/plugins/DDShashlikEndcap.h"
#include "Geometry/HGCalCommonData/plugins/DDShashlikNoTaperSupermodule.h"
#include "Geometry/HGCalCommonData/plugins/DDShashlikNoTaperEndcap.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalEEAlgo, "hgcal:DDHGCalEEAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalHEAlgo, "hgcal:DDHGCalHEAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDShashlikModule, "shashlik:DDShashlikModule");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDShashlikSupermodule, "shashlik:DDShashlikSupermodule");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDShashlikEndcap, "shashlik:DDShashlikEndcap");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDShashlikNoTaperSupermodule, "shashlik:DDShashlikNoTaperSupermodule");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDShashlikNoTaperEndcap, "shashlik:DDShashlikNoTaperEndcap");
