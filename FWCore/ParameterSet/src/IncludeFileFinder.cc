#include "FWCore/ParameterSet/interface/IncludeFileFinder.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "PluginManager/ModuleDescriptor.h"
#include "PluginManager/ModuleCache.h"
#include "PluginManager/Module.h"
#include <iostream>

using namespace seal;
using std::string;
using std::vector;
using std::pair;

namespace edm {
  namespace pset {

    IncludeFileFinder::IncludeFileFinder()
    : thePluginManager(seal::PluginManager::get()),
      theLibraryMap()
    {
      thePluginManager->initialise();
      // map every module to its library.  Code copied from SealPluginDump
      for (PluginManager::DirectoryIterator moduleCache = thePluginManager->beginDirectories ();
           moduleCache != thePluginManager->endDirectories (); ++moduleCache)
      {
        for (ModuleCache::Iterator module = (*moduleCache)->begin (); 
             module != (*moduleCache)->end (); ++module)
        {
          string libraryName = (*module)->libraryName().name();
          ModuleDescriptor * moduleDescriptor=(*module)->cacheRoot();
          for (unsigned i=0; i < moduleDescriptor->children(); ++i)
          {
            string moduleClass = moduleDescriptor->child(i)->token(1);

            theLibraryMap[moduleClass] = libraryName;
          }
        }
      }
    }


    edm::FileInPath IncludeFileFinder::find(const string & moduleClass, 
                                            const string & moduleLabel)
    {
      FileInPath result;
      string libraryName = libraryOf(moduleClass);
      string name = stripHeader(libraryName);
      name = stripTrailer(name);
      // parse around capital letters
      DomainPackagePair parsings
        = twoWordsFrom(name);

      for(DomainPackagePair::const_iterator wordPairItr = parsings.begin();
          wordPairItr != parsings.end(); ++wordPairItr)
      {
        string path = wordPairItr->first + "/" + wordPairItr->second + "/data/"
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


    string IncludeFileFinder::libraryOf(const string & moduleClass)
    {
      std::map<string, string>::const_iterator mapItr 
        = theLibraryMap.find(moduleClass);
      if(mapItr == theLibraryMap.end())
      {
        throw edm::Exception(errors::Configuration, "IncludeFileFinder")
          << "Cannot find " << moduleClass << " in the SEAL Plugin list";
      }
      return mapItr->second;
    }


    string IncludeFileFinder::stripHeader(const string & libraryName)
    {
      // we expect a filename like "libMyDomainYourPackage.so"
      // first strip off the "lib"
      string result;
      if(libraryName.substr(0, 3) != "lib")
      {
        std::cerr << "Strange library name in IncludeFileFinder: "
             << libraryName << ", doesn't start with 'lib'";
        // try to continue
        result = libraryName;
      }
      else 
      {
        // strip it off
        result = libraryName.substr(3, libraryName.size());
      }
      return result;
    }


    string IncludeFileFinder::stripTrailer(const string & libraryName)
    {
      string result;
      unsigned  pos = libraryName.find(".", 0);
      if(pos == string::npos)
      {
        std::cerr << "Strange library name in IncludeFileFinder: "
             << libraryName << ", no dots";
        // try to continue
        result = libraryName;
      }
      else
      {
        // strip it off
        result = libraryName.substr(0, pos);
      }
      return result;
    }


    IncludeFileFinder::DomainPackagePair
    IncludeFileFinder::twoWordsFrom(const string & name)
    {
      DomainPackagePair result;
      for(unsigned i = 1; i < name.size()-1 ; ++i) 
      {
        // split words at capital letters
        if(isupper(name[i]))
        {
          string firstWord = name.substr(0, i);
          string secondWord = name.substr(i, name.size()-i);
          result.push_back(pair<string, string>(firstWord, secondWord));
        }
      }
      return result;
    }      

  }
} 


