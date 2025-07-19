
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerClusterAlgoFactory.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalImagingAlgo.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgo.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"

DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HGCalImagingAlgo, "Imaging");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HGCalSiCLUEAlgo, "SiCLUE");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HGCalSciCLUEAlgo, "SciCLUE");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HFNoseCLUEAlgo, "HFNoseCLUE");
