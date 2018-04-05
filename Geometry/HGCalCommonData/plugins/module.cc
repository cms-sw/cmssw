#include "Geometry/HGCalCommonData/plugins/DDHGCalEEAlgo.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalHEAlgo.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalCell.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalWaferAlgo.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalWafer.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalWafer8.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalModule.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalModuleAlgo.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalTBModule.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalNoTaperEndcap.h"
#include "Geometry/HGCalCommonData/plugins/DDAHcalModuleAlgo.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"

DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalEEAlgo, "hgcal:DDHGCalEEAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalHEAlgo, "hgcal:DDHGCalHEAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalCell,   "hgcal:DDHGCalCell");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalWaferAlgo, "hgcal:DDHGCalWaferAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalWafer, "hgcal:DDHGCalWafer");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalWafer8, "hgcal:DDHGCalWafer8");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalModule, "hgcal:DDHGCalModule");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalModuleAlgo, "hgcal:DDHGCalModuleAlgo");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalTBModule, "hgcal:DDHGCalTBModule");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDHGCalNoTaperEndcap, "hgcal:DDHGCalNoTaperEndcap");
DEFINE_EDM_PLUGIN (DDAlgorithmFactory, DDAHcalModuleAlgo, "hgcal:DDAHcalModuleAlgo");
