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
// $Id: PluginInfo.h,v 1.1.2.2 2007/04/09 18:46:49 chrjones Exp $
//

// system include files
#include <boost/filesystem/path.hpp>

// user include files

// forward declarations
namespace edmplugin {
struct PluginInfo {
  std::string name_;
  boost::filesystem::path loadable_;
};
}
#endif
