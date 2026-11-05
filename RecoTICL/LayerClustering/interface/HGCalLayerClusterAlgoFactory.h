#ifndef RecoLocalCalo_HGCalRecProducers_HGCalLayerClusterAlgoFactory_H
#define RecoLocalCalo_HGCalRecProducers_HGCalLayerClusterAlgoFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalClusteringAlgoBase.h"

typedef edmplugin::PluginFactory<HGCalClusteringAlgoBase*(const edm::ParameterSet&)> HGCalLayerClusterAlgoFactory;

#endif
