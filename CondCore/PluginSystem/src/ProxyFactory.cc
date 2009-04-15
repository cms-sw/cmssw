// -*- C++ -*-
//
// Package:     PluginSystem
// Class  :     ProxyFactory
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Jul 23 19:14:11 EDT 2005
//

// system include files

// user include files
#include "CondCore/PluginSystem/interface/ProxyFactory.h"
#include "CondCore/PluginSystem/interface/DataProxy.h"
//#include <map>
//#include <string>
//#include <iostream>
//
// constants, enums and typedefs
//

// EDM_REGISTER_PLUGINFACTORY(oldcond::ProxyFactory, cond::pluginCategory());
EDM_REGISTER_PLUGINFACTORY(cond::ProxyFactory, cond::pluginCategory());

namespace cond {
  const char*
  pluginCategory()
  {
    return  "CondProxyFactory";
  }
}

