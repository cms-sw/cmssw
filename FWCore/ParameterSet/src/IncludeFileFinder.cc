#include "FWCore/ParameterSet/interface/IncludeFileFinder.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edmplugin;

namespace edm {
  namespace pset {

    IncludeFileFinder::IncludeFileFinder()
    : thePluginManager(0),
      theLibraryMap()
    {
        if(!edmplugin::PluginManager::isAvailable()) {
          edmplugin::PluginManager::configure(edmplugin::standard::config());
        }
        thePluginManager = edmplugin::PluginManager::get();

        typedef edmplugin::PluginManager::CategoryToInfos CatToInfos;
        const std::string kCapability("Capability"); 
        const CatToInfos& catToInfos = thePluginManager->categoryToInfos();
        // map every module to its library.  Code copied from PluginDump
        for (CatToInfos::const_iterator it = catToInfos.begin(), itEnd=catToInfos.end();
             it != itEnd; ++it)
        {
          //NOTE: should filter so only look at the plugins used by the framework, but not looking at 
          // the dictionaries is a good first step
          if( kCapability == it->first) { continue; }
          for (edmplugin::PluginManager::Infos::const_iterator itInfo = it->second.begin(), itInfoEnd = it->second.end(); 
               itInfo != itInfoEnd; ++itInfo)
          {
            typedef std::map<std::string, std::string>::iterator LibMapItr;
            
            std::string moduleClass = itInfo->name_;
            std::pair<LibMapItr,LibMapItr> range = theLibraryMap.equal_range(moduleClass);
            if(range.first == range.second) {
              //the first match is the one to keep
              std::string libraryName = itInfo->loadable_.leaf();
              theLibraryMap.insert(range.first,std::make_pair(moduleClass,libraryName));
            }
          }
        }
    }

    edm::FileInPath IncludeFileFinder::find(const std::string & moduleClass, 
                                            const std::string & moduleLabel)
    {
      FileInPath result;
      std::string libraryName = libraryOf(moduleClass);
      std::string name = stripHeader(libraryName);
      name = stripTrailer(name);
      // parse around capital letters
      DomainPackagePair parsings
        = twoWordsFrom(name);

      for(DomainPackagePair::const_iterator wordPairItr = parsings.begin(), wordPairItrEnd = parsings.end();
          wordPairItr != wordPairItrEnd; ++wordPairItr)
      {
        std::string path = wordPairItr->first + "/" + wordPairItr->second + "/data/"
                      + moduleLabel + ".cfi";
        // see if there's a file with this name
        try
        {
          result = FileInPath(path);
          // no exception?  Good.  We have a winner.
          return result;
        }
        catch(edm::Exception & e)
        {
          // keep trying
        }
      }
      // nothing.  Throw
      throw edm::Exception(errors::Configuration, "IncludeFileFinder")
        << "Could not find file " << moduleLabel + ".cfi" << " in "
        << name;
       // just to suppress compiler warnings
       return result;
    }


    std::string IncludeFileFinder::libraryOf(const std::string & moduleClass)
    {
      std::map<std::string, std::string>::const_iterator mapItr 
        = theLibraryMap.find(moduleClass);
      if(mapItr == theLibraryMap.end())
      {
        throw edm::Exception(errors::Configuration, "IncludeFileFinder")
          << "Cannot find " << moduleClass << " in the Edm Plugin list";
      }
      return mapItr->second;
    }


    std::string IncludeFileFinder::stripHeader(const std::string & libraryName)
    {
      // We expect a filename like "libMyDomainYourPackage.so"
      // or pluginMyDomainYourPackage.so
      // first strip off the "lib" or "plugin".
      std::string result;
      if(libraryName.substr(0, 3) == "lib")
      {
        // strip it off
        result = libraryName.substr(3, libraryName.size());
      }
      else if(libraryName.substr(0, 6) == "plugin")
      {
        result = libraryName.substr(6, libraryName.size());
      }
      else
      {
        edm::LogError("Configuration")
             << "Strange library name in IncludeFileFinder: "
             << libraryName << ", doesn't start with 'lib' or 'plugin'";
        // try to continue
        result = libraryName;
      }
      return result;
    }


    std::string IncludeFileFinder::stripTrailer(const std::string & libraryName)
    {
      std::string result;
      unsigned  pos = libraryName.find(".", 0);
      if(pos == std::string::npos)
      {
        edm::LogError("Configuration")
            << "Strange library name in IncludeFileFinder: "
            << libraryName << ", no dots";
        // try to continue
        result = libraryName;
      }
      else
      {
        // strip it off
        result = libraryName.substr(0, pos);
      }

      pos = result.length();
      // now get rid of the word "Plugins"
      if(result.substr(pos-7, pos) == "Plugins")
      {
        result = result.substr(0, pos-7);
      }
      return result;
    }


    IncludeFileFinder::DomainPackagePair
    IncludeFileFinder::twoWordsFrom(const std::string & name)
    {
      DomainPackagePair result;
      for(unsigned i = 1; i < name.size()-1 ; ++i) 
      {
        // split words at capital letters
        if(isupper(name[i]))
        {
          std::string firstWord = name.substr(0, i);
          std::string secondWord = name.substr(i, name.size()-i);
          result.push_back(std::pair<std::string, std::string>(firstWord, secondWord));
        }
      }
      return result;
    }      

  }
} 


