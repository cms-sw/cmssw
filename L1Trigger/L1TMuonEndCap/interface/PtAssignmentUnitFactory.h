#ifndef __L1TMUON_PTASSIGNMENTUNITFACTORY_H__
#define __L1TMUON_PTASSIGNMENTUNITFACTORY_H__

// 
// Class: L1TMuon::PtAssignmentUnitFactory
//
// Info: Factory that produces a specified type of PtAssignmentUnit
//
// Author: L. Gray (FNAL)
//

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentUnit.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace L1TMuon {
  typedef 
    edmplugin::PluginFactory<PtAssignmentUnit*(const edm::ParameterSet&)>
    PtAssignmentUnitFactory;
}

#endif
