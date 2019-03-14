#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalLayerClusterAlgoFactory.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalClusteringAlgoBase.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalImagingAlgo.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalCLUEAlgo.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h"
#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"

EDM_REGISTER_VALIDATED_PLUGINFACTORY(HGCalLayerClusterAlgoFactory, "HGCalLayerClusterAlgoFactory");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HGCalImagingAlgo, "Imaging");
DEFINE_EDM_VALIDATED_PLUGIN(HGCalLayerClusterAlgoFactory, HGCalCLUEAlgo, "CLUE");
