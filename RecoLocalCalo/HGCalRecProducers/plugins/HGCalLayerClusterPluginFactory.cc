#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerClusterAlgoFactory.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalImagingAlgo.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCLUEAlgo.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(HGCalLayerClusterAlgoFactory, "HGCalLayerClusterAlgoFactory");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HGCalImagingAlgo, "Imaging");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HGCalCLUEAlgo, "CLUE");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HFNoseCLUEAlgo, "HFNoseCLUE");
