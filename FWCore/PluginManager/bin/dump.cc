//<<<<<< INCLUDES                                                       >>>>>>


#include <iostream>
#include <utility>
#include <cstdlib>
#include <string>
#include <set>
#include <boost/program_options.hpp>

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"


using namespace edmplugin;


int main (int argc, char **argv)
{
  using namespace boost::program_options;
  
  static const char* const kFilesOpt = "files";
  static const char* const kFilesCommandOpt = "files,f";
  static const char* const kAllFilesOpt = "all_files";
  static const char* const kAllFilesCommandOpt = "all_files,a";
  static const char* const kHelpOpt = "help";
  static const char* const kHelpCommandOpt = "help,h";
  
  std::string descString(argv[0]);
  descString += " [options]";
  descString += "\nAllowed options";
  options_description desc(descString);
  desc.add_options()
    (kHelpCommandOpt, "produce help message")
    (kFilesCommandOpt
     , "list the file from which a plugin will come")
    (kAllFilesCommandOpt
     , "list all the files to which a plugin is registered")
    //(kAllCommandOpt,"when no paths given, try to update caches for all known directories [default is to only scan the first directory]")
    ;
  
  variables_map vm;
  try {
    store(command_line_parser(argc,argv).options(desc).run(),vm);
    notify(vm);
  } catch(const error& iException) {
    std::cerr <<iException.what();
    return 1;
  }
  
  if(vm.count(kHelpOpt)) {
    std::cout << desc <<std::endl;
    return 0;
  }
  
  bool printFiles = false;
  if(vm.count(kFilesOpt) ) {
    printFiles = true;
  }
  
  bool printAllFiles = false;
  if(vm.count(kAllFilesOpt) ) {
    printFiles = true;
    printAllFiles = true;
  }
  
  int returnValue = EXIT_SUCCESS;
  try {
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
          std::cout <<"  "<<pluginName<<std::endl;
          if(printFiles) {
            std::cout <<"   "<<itInfo->loadable_.string()<<std::endl;
          }
          prevPluginName=pluginName;
        }
        else if(printAllFiles) {
          std::cout <<"   "<<itInfo->loadable_.string()<<std::endl;
        }
      }
    }
  }catch(std::exception& iException) {
    std::cerr <<"Caught exception "<<iException.what()<<std::endl;
    returnValue = 1;
  }

    return returnValue;
}
