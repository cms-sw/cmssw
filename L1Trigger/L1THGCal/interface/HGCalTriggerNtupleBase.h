#ifndef __L1Trigger_L1THGCal_HGCalTriggerNtupleBase_h__
#define __L1Trigger_L1THGCal_HGCalTriggerNtupleBase_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include"TTree.h"


class HGCalTriggerNtupleBase
{
    public:
        HGCalTriggerNtupleBase(const edm::ParameterSet& conf) {};
        virtual ~HGCalTriggerNtupleBase(){};	
        virtual void initialize(TTree& , const edm::ParameterSet&, edm::ConsumesCollector&& ) = 0;
        virtual void fill(const edm::Event& , const edm::EventSetup& ) = 0;

    protected:
        virtual void clear() = 0;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< HGCalTriggerNtupleBase* (const edm::ParameterSet&) > HGCalTriggerNtupleFactory;


#endif
