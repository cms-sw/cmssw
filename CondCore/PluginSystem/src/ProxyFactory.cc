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
// $Id: ProxyFactory.cc,v 1.7 2007/08/23 11:33:08 xiezhen Exp $
//

// system include files

// user include files
#include "CondCore/PluginSystem/interface/ProxyFactory.h"
//#include <map>
//#include <string>
//#include <iostream>
//
// constants, enums and typedefs
//

EDM_REGISTER_PLUGINFACTORY(oldcond::ProxyFactory, cond::pluginCategory());

namespace cond {
  const char*
  pluginCategory()
  {
    return  "CondProxyFactory";
  }
}

