//<<<<<< INCLUDES                                                       >>>>>>

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <utility>
#include <cstdlib>
#include <string>
#include <set>

using namespace edmplugin;


int main (int argc, char **argv)
{
  int returnValue = EXIT_SUCCESS;
  try {
    if (argc == 1) {
      //dump all know plugins
      PluginManager::configure(standard::config());
      
      typedef edmplugin::PluginManager::CategoryToInfos CatToInfos;

      const CatToInfos& catToInfos = edmplugin::PluginManager::get()->categoryToInfos();
      // map every module to its library.  Code copied from EdmPluginDump
      for (CatToInfos::const_iterator it = catToInfos.begin(), itEnd=catToInfos.end();
           it != itEnd; ++it)
      {
        std::cout <<"Category "<<it->first<<":"<<std::endl;
        std::string prevPluginName;
        for (edmplugin::PluginManager::Infos::const_iterator itInfo = it->second.begin(), itInfoEnd = it->second.end(); 
             itInfo != itInfoEnd; ++itInfo)
        {
          std::string pluginName = itInfo->name_;
          if(pluginName != prevPluginName) {
            std::cout <<"  "<<pluginName << std::endl;
            prevPluginName=pluginName;
          }
        }
      }
    } else {
      std::cerr <<"no arguments allowed"<<std::endl;
    }
  }catch(std::exception& iException) {
    std::cerr <<"Caught exception "<<iException.what()<<std::endl;
    returnValue = 1;
  }

    return returnValue;
}
