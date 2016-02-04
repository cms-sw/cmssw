#ifndef __L1TMUON_PTREFINEMENTUNITFACTORY_H__
#define __L1TMUON_PTREFINEMENTUNITFACTORY_H__

// 
// Class: L1TMuon::PtRefinementUnitFactory
//
// Info: Factory that produces a specified type of PtRefinementUnit
//
// Author: L. Gray (FNAL)
//

#include "L1Trigger/L1TMuonEndCap/interface/PtRefinementUnit.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace L1TMuon {
  typedef 
    edmplugin::PluginFactory<PtRefinementUnit*(const edm::ParameterSet&)>
    PtRefinementUnitFactory;
}

#endif
