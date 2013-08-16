#ifndef FWCore_PluginManager_standard_h
#define FWCore_PluginManager_standard_h
// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     standard
// 
/**\class standard standard.h FWCore/PluginManager/interface/standard.h

 Description: namespace which holds the standard configuration information

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Apr  7 17:06:34 EDT 2007
//

// system include files
#include <string>
#include <boost/filesystem/path.hpp>

// user include files
#include "FWCore/PluginManager/interface/PluginManager.h"

// forward declarations

namespace edmplugin {
  namespace standard {

    PluginManager::Config config();
    
    const boost::filesystem::path& cachefileName();
    
    const std::string& pluginPrefix();
  }
}
#endif
