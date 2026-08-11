#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"
#include "RecoTICL/LayerClustering/interface/HGCalLayerClusterAlgoFactory.h"
#include "BarrelCLUEAlgo.h"

DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, EBCLUEAlgo, "EBCLUE");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HBCLUEAlgo, "HBCLUE");
