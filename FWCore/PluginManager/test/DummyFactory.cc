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
// $Id: DummyFactory.cc,v 1.1.2.2 2007/04/12 09:12:25 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/PluginManager/test/DummyFactory.h"

namespace testedmplugin {
  DummyBase::~DummyBase() {}
}

EDM_REGISTER_PLUGINFACTORY(testedmplugin::DummyFactory,"Test Dummy")
