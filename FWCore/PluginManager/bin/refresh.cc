#include "FWCore/PluginManager/interface/CacheParser.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
#include "FWCore/PluginManager/interface/PluginFactoryManager.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/PluginManager/interface/standard.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <boost/bind.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/mem_fn.hpp>
#include <boost/program_options.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <utility>

#include <sys/wait.h>

using namespace edmplugin;

// We process DSOs by forking and processing them in the child, in bunches of 
// PER_PROCESS_DSO. Once a bunch has been processed we exit the child and
// spawn a new one.
// Due to the different behavior of the linker on different platforms, we process
// a different number of DSOs per process.
// For macosx the cost is to actually resolve weak symbols and it grows with
// the number of libraries as well, so we try to process not too many libraries.
// In linux the main cost is to load the library itself, so we try to load as 
// many as reasonable in a single process to avoid having to reload them in 
// the subsequent process. Obviuously keeping the PER_PROCESS_DSO low keeps the
// memory usage low as well.
#ifdef __APPLE__
#define PER_PROCESS_DSO 20
#else
#define PER_PROCESS_DSO 200
#endif

namespace std {
  ostream& operator<<(std::ostream& o, vector<std::string> const& iValue) {
    std::string sep("");
    std::string commaSep(",");
    for(std::vector<std::string>::const_iterator it = iValue.begin(), itEnd = iValue.end();
        it != itEnd;
        ++it) {
      o << sep << *it;
      sep = commaSep;
    }
    return o;
  }
}
namespace {
  struct Listener {
    typedef edmplugin::CacheParser::NameAndType NameAndType;
    typedef edmplugin::CacheParser::NameAndTypes NameAndTypes;

    void newFactory(edmplugin::PluginFactoryBase const* iBase) {
      iBase->newPluginAdded_.connect(boost::bind(boost::mem_fn(&Listener::newPlugin), this, _1, _2));
    }
    void newPlugin(std::string const& iCategory, edmplugin::PluginInfo const& iInfo) {
      nameAndTypes_.push_back(NameAndType(iInfo.name_, iCategory));
    }

    NameAndTypes nameAndTypes_;
  };
}
int main (int argc, char **argv) {
  using namespace boost::program_options;

  static char const* const kPathsOpt = "paths";
  static char const* const kPathsCommandOpt = "paths,p";
  //static char const* const kAllOpt = "all";
  //static char const* const kAllCommandOpt = "all,a";
  static char const* const kHelpOpt = "help";
  static char const* const kHelpCommandOpt = "help,h";

  std::string descString(argv[0]);
  descString += " [options] [[--";
  descString += kPathsOpt;
  descString += "] path [path]] \nAllowed options";
  options_description desc(descString);
  std::string defaultDir(".");
  std::vector<std::string> defaultDirList = edmplugin::standard::config().searchPath();
  if(!defaultDirList.empty()) {
    defaultDir = defaultDirList[0];
  }
  desc.add_options()
    (kHelpCommandOpt, "produce help message")
    (kPathsCommandOpt, value<std::vector<std::string> >()->default_value(
                          std::vector<std::string>(1, defaultDir))
     , "a directory or a list of files to scan")
    //(kAllCommandOpt, "when no paths given, try to update caches for all known directories [default is to only scan the first directory]")
    ;

  positional_options_description p;
  p.add(kPathsOpt, -1);

  variables_map vm;
  try {
    store(command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    notify(vm);
  } catch(error const& iException) {
    std::cerr << iException.what();
    return 1;
  }

  if(vm.count(kHelpOpt)) {
    std::cout << desc << std::endl;
    return 0;
  }

  using boost::filesystem::path;

  /*if(argc == 1) {
    std::cerr << "Requires at least one argument.  Please pass either one directory or a list of files (all in the same directory)." << std::endl;
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
      if (requestedPaths.size() > 1) {
        std::cerr << "if a directory is given then only one argument is allowed" << std::endl;
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
        if(stringName.substr(0, kPluginPrefix.size()) != kPluginPrefix) {
          continue;
        }

        if(last_write_time(filename) > cacheLastChange) {
          files.push_back(stringName);
        }
      }
    } else {
      //we have files
      directory = directory.branch_path();
      for(std::vector<std::string>::iterator it = requestedPaths.begin(), itEnd = requestedPaths.end();
          it != itEnd; ++it) {
        boost::filesystem::path f(*it);
        if (!exists(f)) {
          std::cerr << "the file '" << f.native_file_string() << "' does not exist" << std::endl;
          return 1;
        }
        if (is_directory(f)) {
          std::cerr << "either one directory or a list of files are allowed as arguments" << std::endl;
          return 1;
        }
        if(directory != f.branch_path()) {
          std::cerr << "all files must have be in the same directory (" << directory.native_file_string() << ")\n"
          " the file " << f.native_file_string() << " does not." << std::endl;
        }
        files.push_back(f.leaf());
      }
    }

    path cacheFile(directory);
    cacheFile /= edmplugin::standard::cachefileName();//path(s_cacheFile);

    CacheParser::LoadableToPlugins old;
    if(exists(cacheFile)) {
      std::ifstream cf(cacheFile.native_file_string().c_str());
      if(!cf) {
        cms::Exception("FailedToOpen") << "unable to open file '" << cacheFile.native_file_string() << "' for reading even though it is present.\n"
        "Please check permissions on the file.";
      }
      CacheParser::read(cf, old);
    }


    //load each file and 'listen' to which plugins are loaded
    Listener listener;
    edmplugin::PluginFactoryManager* pfm =  edmplugin::PluginFactoryManager::get();
    pfm->newFactory_.connect(boost::bind(boost::mem_fn(&Listener::newFactory), &listener, _1));
    edm::for_all(*pfm, boost::bind(boost::mem_fn(&Listener::newFactory), &listener, _1));

    // We open the cache file before forking so that all the children will
    // use it.
    std::string temporaryFilename = (cacheFile.native_file_string() + ".tmp");
    std::ofstream cf(temporaryFilename.c_str());
    if(!cf) {
      cms::Exception("FailedToOpen") << "unable to open file '" 
          << temporaryFilename << "' for writing.\n"
      "Please check permissions on the file.";
    }
    // Sort the files so that they are loaded "by subsystem", hopefully meaning
    // they share more dependencies.
    std::sort(files.begin(), files.end());
    
    for(size_t fi = 0, fe = files.size(); fi < fe; fi += PER_PROCESS_DSO)
    {
      CacheParser::LoadableToPlugins ltp;
      pid_t worker = fork();
      if (worker == 0)
      {
        // This the child process.
        // We load the DSO and find out its plugins, write to the cache 
        // stream and exit, leaving the parent to spawn a new proces.
        size_t ci = PER_PROCESS_DSO;
        while (ci && fi != fe)
        {
          path loadableFile(directory);                                                                                                                                                                                                                                          
          loadableFile /= (files[fi]);
          listener.nameAndTypes_.clear();

          returnValue = 0;

          try {
           try {
               edmplugin::SharedLibrary lib(loadableFile);
               //PluginCapabilities is special, the plugins do not call it.  Instead, for each shared library load
               // we need to ask it to try to find plugins
               PluginCapabilities::get()->tryToFind(lib);
               ltp[files[fi]] = listener.nameAndTypes_;
            
            } catch(cms::Exception const& iException) {
               if(iException.category() == "PluginLibraryLoadError") {
                  std::cerr << "Caught exception " << iException.what() << " will ignore " << files[fi] << " and continue." << std::endl;
               } else {
                  throw;
               }
            }
          }catch(std::exception& iException) {
            std::cerr << "Caught exception " << iException.what() << std::endl;
            exit(1);
          }
          ++fi;
          --ci;
        }
        CacheParser::write(ltp, cf);
        cf << std::flush;
        _exit(0);
      }
      else
      {
        // Throw if any of the child died with non 0 status.
        int status = 0;
        waitpid(worker, &status, 0);
        if (WIFEXITED(status) == true && status != 0)
        {
          std::cerr << "Error while processing." << std::endl;
          exit(status);
        }
      }
    }
    
    cf << std::flush;
    
    // We read the new cache and we merge it with the old one.
    CacheParser::LoadableToPlugins ltp;
    std::ifstream icf(temporaryFilename.c_str());
    if(!icf) {
      cms::Exception("FailedToOpen") << "unable to open file '" << temporaryFilename.c_str() << "' for reading even though it is present.\n"
      "Please check permissions on the file.";
    }
    CacheParser::read(icf, ltp);
    
    for (CacheParser::LoadableToPlugins::iterator itFile = ltp.begin() ;
         itFile != ltp.end() ;
         ++itFile)
    {
      old[itFile->first] = itFile->second; 
    }
    
    // If required, we remove the plugins which are missing. Notice that old is
    // now the most updated copy of the cache.
    if(removeMissingFiles) {
      for(CacheParser::LoadableToPlugins::iterator itFile = old.begin();
          itFile != old.end();
          /*don't advance the iterator here because it may have become invalid */) {
        path loadableFile(directory);
        loadableFile /= (itFile->first);
        if(not exists(loadableFile)) {
          std::cout << "removing file '" << temporaryFilename.c_str() << "'" << std::endl;
          CacheParser::LoadableToPlugins::iterator itToItemBeingRemoved = itFile;
          //advance the iterator while it is still valid
          ++itFile;
          old.erase(itToItemBeingRemoved);
        } else {
          //since we are not advancing the iterator in the for loop, do it here
          ++itFile;
        }
      }
    }
  
    // We finally write the final cache.
    std::ofstream fcf(temporaryFilename.c_str());
    if(!fcf) {
      cms::Exception("FailedToOpen") << "unable to open file '" << temporaryFilename.c_str() << "' for writing.\n"
      "Please check permissions on the file.";
    }
    CacheParser::write(old, fcf);
    rename(temporaryFilename.c_str(), cacheFile.native_file_string().c_str());  
  } catch(std::exception& iException) {
    std::cerr << "Caught exception " << iException.what() << std::endl;
    returnValue = 1;
  }

  return returnValue;
}
