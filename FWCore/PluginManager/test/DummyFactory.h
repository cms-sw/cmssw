#ifndef FWCore_PluginManager_DummyFactory_h
#define FWCore_PluginManager_DummyFactory_h
// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     DummyFactory
// 
/**\class DummyFactory DummyFactory.h FWCore/PluginManager/interface/DummyFactory.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Apr  6 15:26:43 EDT 2007
// $Id: DummyFactory.h,v 1.1.2.1 2007/04/09 18:46:51 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/PluginManager/interface/PluginFactory.h"

// forward declarations
namespace testedmplugin {
  struct DummyBase {
    virtual ~DummyBase();
    virtual int value() const = 0;
  };


  typedef edmplugin::PluginFactory<DummyBase*(void)> DummyFactory;
}

#endif
