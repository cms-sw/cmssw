
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/bind.hpp>
#include <boost/mem_fn.hpp>

#include <boost/program_options.hpp>

#include <fstream>
#include <iostream>
#include <utility>
#include <cstdlib>
#include <string>
#include <set>
#include <algorithm>

#include "FWCore/PluginManager/interface/PluginFactoryManager.h"
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/PluginManager/interface/CacheParser.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"

#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "FWCore/PluginManager/interface/standard.h"

using namespace edmplugin;

namespace std {
ostream& operator<<(std::ostream& o, const vector<std::string>& iValue) {
  std::string sep("");
  std::string commaSep(",");
  for(std::vector<std::string>::const_iterator it=iValue.begin(), itEnd=iValue.end();
      it != itEnd;
      ++it) {    
    o <<sep<<*it;
    sep = commaSep;
  }
  return o;
}
}
namespace {
  struct Listener {
    typedef edmplugin::CacheParser::NameAndType NameAndType;
    typedef edmplugin::CacheParser::NameAndTypes NameAndTypes;
    
    void newFactory(const edmplugin::PluginFactoryBase* iBase) {
      iBase->newPluginAdded_.connect(boost::bind(boost::mem_fn(&Listener::newPlugin),this,_1,_2));
    }
    void newPlugin(const std::string& iCategory, const edmplugin::PluginInfo& iInfo) {
      nameAndTypes_.push_back(NameAndType(iInfo.name_,iCategory));
    }
    
    NameAndTypes nameAndTypes_;
  };
}
int main (int argc, char **argv)
{
  using namespace boost::program_options;
  
  static const char* const kPathsOpt = "paths";
  static const char* const kPathsCommandOpt = "paths,p";
  //static const char* const kAllOpt = "all";
  //static const char* const kAllCommandOpt = "all,a";
  static const char* const kHelpOpt = "help";
  static const char* const kHelpCommandOpt = "help,h";
  
  std::string descString(argv[0]);
  descString += " [options] [[--";
  descString += kPathsOpt;
  descString += "] path [path]] \nAllowed options";
  options_description desc(descString);
  std::string defaultDir(".");
  std::vector<std::string> defaultDirList = edmplugin::standard::config().searchPath();
  if( not defaultDirList.empty() ) {
    defaultDir = defaultDirList[0];
  }
  desc.add_options()
    (kHelpCommandOpt, "produce help message")
    (kPathsCommandOpt,value<std::vector<std::string> >()->default_value(
                          std::vector<std::string>(1,defaultDir))
     , "a directory or a list of files to scan")
    //(kAllCommandOpt,"when no paths given, try to update caches for all known directories [default is to only scan the first directory]")
    ;
  
  positional_options_description p;
  p.add(kPathsOpt, -1);
  
  variables_map vm;
  try {
    store(command_line_parser(argc,argv).options(desc).positional(p).run(),vm);
    notify(vm);
  } catch(const error& iException) {
    std::cerr <<iException.what();
    return 1;
  }
  
  if(vm.count(kHelpOpt)) {
    std::cout << desc <<std::endl;
    return 0;
  }
  
  
  using boost::filesystem::path;
  
  /*if(argc ==1) {
    std::cerr <<"Requires at least one argument.  Please pass either one directory or a list of files (all in the same directory)."<<std::endl;
    return 1;
  } */

  int returnValue = EXIT_SUCCESS;

  try {
    std::vector<std::string> requestedPaths(vm[kPathsOpt].as<std::vector<std::string> >());
    
    //first find the directory and create a list of files to look at in that directory
    path directory(requestedPaths[0]);
    std::vector<std::string> files;
    bool removeMissingFiles = false;
    if(boost::filesystem::is_directory(directory)) {
      if (requestedPaths.size()>1) {
        std::cerr <<"if a directory is given then only one argument is allowed"<<std::endl;
        return 1;
      }
      
      //if asked to look at whole directory, then we can also remove missing files
      removeMissingFiles = true;
      
      boost::filesystem::directory_iterator       file (directory);
      boost::filesystem::directory_iterator       end;

      path cacheFile(directory);
      cacheFile /= standard::cachefileName();

      std::time_t cacheLastChange(0);
      if(exists(cacheFile)) {
        cacheLastChange = last_write_time(cacheFile);
      }
      for (; file != end; ++file)
      {

        path  filename (*file);
        path shortName(file->leaf());
        std::string stringName = shortName.string();
        
        static std::string kPluginPrefix(standard::pluginPrefix());
        if (stringName.size() < kPluginPrefix.size()) {
          continue;
        }
        if(stringName.substr(0,kPluginPrefix.size()) != kPluginPrefix) {
          continue;
        }

        if(last_write_time(filename) > cacheLastChange) {
          files.push_back(stringName);
        }
      }
    } else {
      //we have files
      directory = directory.branch_path();
      for(std::vector<std::string>::iterator it=requestedPaths.begin(), itEnd=requestedPaths.end();
          it != itEnd; ++it) {
        boost::filesystem::path f(*it);
        if ( not exists(f) ) {
          std::cerr <<"the file '"<<f.native_file_string()<<"' does not exist"<<std::endl;
          return 1;
        }
        if (is_directory(f) ) {
          std::cerr <<"either one directory or a list of files are allowed as arguments"<<std::endl;
          return 1;
        }
        if(directory != f.branch_path()) {
          std::cerr <<"all files must have be in the same directory ("<<directory.native_file_string()<<")\n"
          " the file "<<f.native_file_string()<<" does not."<<std::endl;
        }
        files.push_back(f.leaf());
      }
    }

    path cacheFile(directory);
    cacheFile /= edmplugin::standard::cachefileName();//path(s_cacheFile);

    CacheParser::LoadableToPlugins ltp;
    if(exists(cacheFile) ) {
      std::ifstream cf(cacheFile.native_file_string().c_str());
      if(!cf) {
        cms::Exception("FailedToOpen")<<"unable to open file '"<<cacheFile.native_file_string()<<"' for reading even though it is present.\n"
        "Please check permissions on the file.";
      }
      CacheParser::read(cf, ltp);
    }
    
    
    //load each file and 'listen' to which plugins are loaded
    Listener listener;
    edmplugin::PluginFactoryManager* pfm =  edmplugin::PluginFactoryManager::get();
    pfm->newFactory_.connect(boost::bind(boost::mem_fn(&Listener::newFactory),&listener,_1));
    edm::for_all(*pfm, boost::bind(boost::mem_fn(&Listener::newFactory),&listener,_1));
    
    for(std::vector<std::string>::iterator itFile = files.begin();
        itFile != files.end();
        ++itFile) {

      path loadableFile(directory);
      loadableFile /=(*itFile);
      listener.nameAndTypes_.clear();
      try {
         edmplugin::SharedLibrary lib(loadableFile);
      
         //PluginCapabilities is special, the plugins do not call it.  Instead, for each shared library load
         // we need to ask it to try to find plugins
         PluginCapabilities::get()->tryToFind(lib);
      
         ltp[*itFile]=listener.nameAndTypes_;
      } catch(const cms::Exception& iException) {
         if(iException.category() == "PluginLibraryLoadError") {
            std::cerr <<"Caught exception "<<iException.what()<<" will ignore "<<*itFile<<" and continue."<<std::endl;
         } else {
            throw;
         }
      }
    }
    
    if(removeMissingFiles) {
      for(CacheParser::LoadableToPlugins::iterator itFile = ltp.begin();
          itFile != ltp.end();
          /*don't advance the iterator here because it may have become invalid */) {
        path loadableFile(directory);
        loadableFile /=(itFile->first);
        if(not exists(loadableFile)) {
          std::cout <<"removing file '"<<loadableFile.native_file_string()<<"'"<<std::endl;
          CacheParser::LoadableToPlugins::iterator itToItemBeingRemoved = itFile;
          //advance the iterator while it is still valid
          ++itFile;
          ltp.erase(itToItemBeingRemoved);
        } else {
          //since we are not advancing the iterator in the for loop, do it here
          ++itFile;
        }
      }
      //now get rid of the items 
    }
    //now write our new results
    std::ofstream cf(cacheFile.native_file_string().c_str());
    if(!cf) {
      cms::Exception("FailedToOpen")<<"unable to open file '"<<cacheFile.native_file_string()<<"' for writing.\n"
      "Please check permissions on the file.";
    }
    CacheParser::write(ltp,cf);
  }catch(std::exception& iException) {
    std::cerr <<"Caught exception "<<iException.what()<<std::endl;
    returnValue = 1;
  }

    return returnValue;
}
