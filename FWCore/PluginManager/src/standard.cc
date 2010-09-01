// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     standard
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Apr  7 17:10:11 EDT 2007
// $Id: standard.cc,v 1.2 2007/04/12 12:51:12 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/PluginManager/interface/standard.h"

namespace edmplugin {
  namespace standard {

    PluginManager::Config config()
  {
      PluginManager::Config returnValue;
      
      const char *path = getenv ("LD_LIBRARY_PATH");
      if (! path) path = "";
      
      std::string spath(path);
      std::string::size_type last=0;
      std::string::size_type i=0;
      std::vector<std::string> paths;
      while( (i=spath.find_first_of(':',last))!=std::string::npos) {
        paths.push_back(spath.substr(last,i-last));
        last = i+1;
        //std::cout <<paths.back()<<std::endl;
      }
      paths.push_back(spath.substr(last,std::string::npos));
      returnValue.searchPath(paths);
      
      return returnValue;
  }
    
    const boost::filesystem::path& cachefileName() {
      static const boost::filesystem::path s_path(".edmplugincache",boost::filesystem::no_check);
      return s_path;
    }
    
    
    const std::string& pluginPrefix() {
      static const std::string s_prefix("plugin");
      return s_prefix;
    }
    
  }
}
