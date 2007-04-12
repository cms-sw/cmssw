
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/bind.hpp>
#include <boost/mem_fn.hpp>

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
#include "FWCore/PluginManager/interface/CacheParser.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"

#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "FWCore/PluginManager/interface/standard.h"
using namespace edmplugin;

static const char s_cacheFile[] = ".edmplugincache";

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
  using boost::filesystem::path;
  if(argc ==1) {
    std::cerr <<"requires at least one argument"<<std::endl;
    return 1;
  }

  int returnValue = EXIT_SUCCESS;

  try {
    //first find the directory and create a list of files to look at in that directory
    path directory(argv[1]);
    std::vector<std::string> files;
    bool removeMissingFiles = false;
    if(boost::filesystem::is_directory(directory)) {
      if (argc >2) {
        std::cerr <<"if a directory is given then only one argument is allowed"<<std::endl;
        return 1;
      }
      
      //if asked to look at who directory, then we can also remove missing files
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
        path shortName(file->leaf(),boost::filesystem::no_check);
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
      for(int index=1; index <argc; ++index) {
        boost::filesystem::path f(argv[index]);
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
    cacheFile /= path(s_cacheFile,boost::filesystem::no_check);

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
    std::for_each(pfm->begin(),
                  pfm->end(),
                  boost::bind(boost::mem_fn(&Listener::newFactory),&listener,_1));
    
    for(std::vector<std::string>::iterator itFile = files.begin();
        itFile != files.end();
        ++itFile) {

      path loadableFile(directory);
      loadableFile /=(*itFile);
      listener.nameAndTypes_.clear();
      edmplugin::SharedLibrary lib(loadableFile);
      
      //PluginCapabilities is special, the plugins do not call it.  Instead, for each shared library load
      // we need to ask it to try to find plugins
      PluginCapabilities::get()->tryToFind(lib);
      
      ltp[*itFile]=listener.nameAndTypes_;
    }
    
    if(removeMissingFiles) {
      for(CacheParser::LoadableToPlugins::iterator itFile = ltp.begin();
          itFile != ltp.end();
          ++itFile) {
        path loadableFile(directory);
        loadableFile /=(itFile->first);
        if(not exists(loadableFile)) {
          std::cout <<"removing file '"<<loadableFile.native_file_string()<<"'"<<std::endl;
          ltp.erase(itFile);
        }
      }
    }
    //now write our new results
    std::ofstream cf(cacheFile.native_file_string().c_str());
    if(!cf) {
      cms::Exception("FailedToOpen")<<"unable to open file '"<<cacheFile.native_file_string()<<"' for writing.\n"
      "Please check permissions on the file.";
    }
    CacheParser::write(ltp,cf);
  }catch(std::exception& iException) {
    std::cout <<"Caught exception "<<iException.what()<<std::endl;
    returnValue = 1;
  }

    return returnValue;
}
