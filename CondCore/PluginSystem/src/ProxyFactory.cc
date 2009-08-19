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
// $Id: ProxyFactory.cc,v 1.6 2007/04/12 21:18:01 chrjones Exp $
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

EDM_REGISTER_PLUGINFACTORY(cond::ProxyFactory, cond::pluginCategory());

namespace cond {
const char*
pluginCategory()
{
  return  "CondProxyFactory";
}
}
