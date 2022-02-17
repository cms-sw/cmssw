#ifndef __L1Trigger_L1THGCal_HGCalAlgoWrapperBase_h__
#define __L1Trigger_L1THGCal_HGCalAlgoWrapperBase_h__

#include "L1Trigger/L1THGCal/interface/HGCalAlgoWrapperBaseT.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"

typedef HGCalAlgoWrapperBaseT<
    std::pair<const std::vector<edm::Ptr<l1t::HGCalCluster>>, const std::vector<std::pair<GlobalPoint, double>>>,
    std::pair<l1t::HGCalMulticlusterBxCollection&, l1t::HGCalClusterBxCollection&>,
    std::pair<const HGCalTriggerGeometryBase* const, const edm::ParameterSet&>>
    HGCalHistoClusteringWrapperBase;

typedef HGCalAlgoWrapperBaseT<std::vector<edm::Ptr<l1t::HGCalTowerMap>>,
                              l1t::HGCalTowerBxCollection,
                              std::pair<const HGCalTriggerGeometryBase* const, const edm::ParameterSet&>>
    HGCalTowerMapsWrapperBase;

typedef HGCalAlgoWrapperBaseT<l1t::HGCalMulticlusterBxCollection,
                              l1t::HGCalMulticlusterBxCollection,
                              std::pair<const HGCalTriggerGeometryBase* const, const edm::ParameterSet&>>
    HGCalStage2FilteringWrapperBase;

typedef HGCalAlgoWrapperBaseT<std::vector<edm::Ptr<l1t::HGCalTriggerCell>>,
                              std::vector<edm::Ptr<l1t::HGCalTriggerCell>>,
                              std::tuple<const HGCalTriggerGeometryBase* const, const unsigned&, const uint32_t&>>
    HGCalStage1TruncationWrapperBase;

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<HGCalHistoClusteringWrapperBase*(const edm::ParameterSet&)>
    HGCalHistoClusteringWrapperBaseFactory;
typedef edmplugin::PluginFactory<HGCalTowerMapsWrapperBase*(const edm::ParameterSet&)> HGCalTowerMapsWrapperBaseFactory;
typedef edmplugin::PluginFactory<HGCalStage2FilteringWrapperBase*(const edm::ParameterSet&)>
    HGCalStage2FilteringWrapperBaseFactory;
typedef edmplugin::PluginFactory<HGCalStage1TruncationWrapperBase*(const edm::ParameterSet&)>
    HGCalStage1TruncationWrapperBaseFactory;

#endif
