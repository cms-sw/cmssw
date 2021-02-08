#ifndef __L1Trigger_L1THGCal_HGCalAlgoWrapperBase_h__
#define __L1Trigger_L1THGCal_HGCalAlgoWrapperBase_h__

#include "L1Trigger/L1THGCal/interface/HGCalAlgoWrapperBaseT.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"

#include "FWCore/Framework/interface/EventSetup.h"

typedef HGCalAlgoWrapperBaseT<
    std::pair<const std::vector<edm::Ptr<l1t::HGCalCluster>>, const std::vector<std::pair<GlobalPoint, double>>>,
    std::pair<l1t::HGCalMulticlusterBxCollection&, l1t::HGCalClusterBxCollection&>,
    std::pair<const edm::EventSetup&, const edm::ParameterSet&>>
    HGCalHistoClusteringWrapperBase;

typedef HGCalAlgoWrapperBaseT<std::vector<edm::Ptr<l1t::HGCalTowerMap>>,
                              l1t::HGCalTowerBxCollection,
                              std::pair<const edm::EventSetup&, const edm::ParameterSet&>>
    HGCalTowerMapsWrapperBase;

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<HGCalHistoClusteringWrapperBase*(const edm::ParameterSet&)>
    HGCalHistoClusteringWrapperBaseFactory;
typedef edmplugin::PluginFactory<HGCalTowerMapsWrapperBase*(const edm::ParameterSet&)> HGCalTowerMapsWrapperBaseFactory;

#endif
