#ifndef __L1Trigger_L1THGCal_HGCalProcessorBase_h__
#define __L1Trigger_L1THGCal_HGCalProcessorBase_h__

#include "L1Trigger/L1THGCal/interface/HGCalProcessorBaseT.h"

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

typedef HGCalProcessorBaseT<HGCalDigiCollection, l1t::HGCalTriggerCellBxCollection> HGCalVFEProcessorBase;
typedef HGCalProcessorBaseT<edm::Handle<l1t::HGCalTriggerCellBxCollection>, l1t::HGCalTriggerCellBxCollection>
    HGCalConcentratorProcessorBase;
typedef HGCalProcessorBaseT<edm::Handle<l1t::HGCalTriggerCellBxCollection>, l1t::HGCalClusterBxCollection>
    HGCalBackendLayer1ProcessorBase;
typedef HGCalProcessorBaseT<edm::Handle<l1t::HGCalClusterBxCollection>, l1t::HGCalMulticlusterBxCollection>
    HGCalBackendLayer2ProcessorBase;
typedef HGCalProcessorBaseT<edm::Handle<l1t::HGCalTriggerCellBxCollection>, l1t::HGCalTowerMapBxCollection>
    HGCalTowerMapProcessorBase;
typedef HGCalProcessorBaseT<edm::Handle<l1t::HGCalTowerMapBxCollection>, l1t::HGCalTowerBxCollection>
    HGCalTowerProcessorBase;

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<HGCalVFEProcessorBase*(const edm::ParameterSet&)> HGCalVFEProcessorBaseFactory;
typedef edmplugin::PluginFactory<HGCalConcentratorProcessorBase*(const edm::ParameterSet&)> HGCalConcentratorFactory;
typedef edmplugin::PluginFactory<HGCalBackendLayer1ProcessorBase*(const edm::ParameterSet&)> HGCalBackendLayer1Factory;
typedef edmplugin::PluginFactory<HGCalBackendLayer2ProcessorBase*(const edm::ParameterSet&)> HGCalBackendLayer2Factory;
typedef edmplugin::PluginFactory<HGCalTowerMapProcessorBase*(const edm::ParameterSet&)> HGCalTowerMapFactory;
typedef edmplugin::PluginFactory<HGCalTowerProcessorBase*(const edm::ParameterSet&)> HGCalTowerFactory;

#endif
