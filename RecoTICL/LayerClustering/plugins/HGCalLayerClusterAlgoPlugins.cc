#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"
#include "RecoTICL/LayerClustering/interface/HGCalImagingAlgo.h"
#include "RecoTICL/LayerClustering/interface/HGCalLayerClusterAlgoFactory.h"
#include "BarrelCLUEAlgo.h"
#include "HGCalCLUEAlgo.h"

DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HGCalImagingAlgo, "Imaging");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HGCalSiCLUEAlgo, "SiCLUE");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HGCalSciCLUEAlgo, "SciCLUE");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HFNoseCLUEAlgo, "HFNoseCLUE");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, EBCLUEAlgo, "EBCLUE");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HBCLUEAlgo, "HBCLUE");
