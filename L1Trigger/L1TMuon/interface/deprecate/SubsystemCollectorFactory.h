#ifndef __L1TMUON_SUBSYSTEMCOLLECTORFACTORY_H__
#define __L1TMUON_SUBSYSTEMCOLLECTORFACTORY_H__

// 
// Class: L1TMuon::SubsystemCollectorFactory
//
// Info: Factory that produces a specified type of SubsystemCollector
//
// Author: L. Gray (FNAL)
//

#include "L1Trigger/L1TMuon/interface/deprecate/SubsystemCollector.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace L1TMuon {
  typedef 
    edmplugin::PluginFactory<SubsystemCollector*(const edm::ParameterSet&)>
    SubsystemCollectorFactory;
}

#endif
