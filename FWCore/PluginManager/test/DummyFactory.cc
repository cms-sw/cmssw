// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     DummyFactory
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Apr  6 15:26:46 EDT 2007
//

// system include files

// user include files
#include "FWCore/PluginManager/test/DummyFactory.h"

namespace testedmplugin {
  DummyBase::~DummyBase() {}
}

EDM_REGISTER_PLUGINFACTORY(testedmplugin::DummyFactory,"Test Dummy");
