#include "Geometry/HGCalCommonData/plugins/DDHGCalEEAlgo.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalHEAlgo.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalWaferAlgo.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalModule.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalModuleAlgo.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalNoTaperEndcap.h"
#include "DetectorDescription/Algorithm/interface/DDAlgorithmFactory.h"

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalEEAlgo, "hgcal:DDHGCalEEAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalHEAlgo, "hgcal:DDHGCalHEAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalWaferAlgo, "hgcal:DDHGCalWaferAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalModule, "hgcal:DDHGCalModule");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalModuleAlgo, "hgcal:DDHGCalModuleAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalNoTaperEndcap, "hgcal:DDHGCalNoTaperEndcap");
