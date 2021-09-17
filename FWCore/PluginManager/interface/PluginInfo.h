#ifndef FWCore_PluginManager_PluginInfo_h
#define FWCore_PluginManager_PluginInfo_h
// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     PluginInfo
//
/**\class PluginInfo PluginInfo.h FWCore/PluginManager/interface/PluginInfo.h

 Description: Holds information about a particular plugin

 Usage:
    

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 19:46:31 EDT 2007
//

// system include files
#include <filesystem>

// user include files

// forward declarations
namespace edmplugin {
  struct PluginInfo {
    std::string name_;
    std::filesystem::path loadable_;
  };
}  // namespace edmplugin
#endif
