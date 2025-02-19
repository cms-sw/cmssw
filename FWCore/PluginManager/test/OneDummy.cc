// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     OneDummy
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Apr  6 15:32:49 EDT 2007
// $Id: OneDummy.cc,v 1.2 2007/04/12 12:51:13 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/PluginManager/test/DummyFactory.h"

namespace testedmplugin {
  struct DummyOne : public DummyBase {
    int value() const {
      return 1;
    }
  };
}

DEFINE_EDM_PLUGIN(testedmplugin::DummyFactory,testedmplugin::DummyOne,"DummyOne");
