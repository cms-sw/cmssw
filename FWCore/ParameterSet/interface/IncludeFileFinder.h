#ifndef ParameterSet_IncludeFileFinder_h
#define ParameterSet_IncludeFileFinder_h

/** Resolves where to look for include files,
    by looking up the module using the SealPluginManager
  */

#include <string>
#include <map>
#include "PluginManager/PluginManager.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

namespace edm {
  namespace pset {

    class IncludeFileFinder
    {
    public:
      IncludeFileFinder();

      /// looks for a file named <label>.cfi in the same directory 
      /// where the class is defined
      FileInPath find(const std::string & moduleClass, 
                      const std::string & moduleLabel);

      /// these are public just for testing purposes
      std::string libraryOf(const std::string & moduleClass);

      /// takes off the 'lib' part of the library name
      static std::string stripHeader(const std::string & libraryName);
      /// takes off the '.so' part of the library name
      static std::string stripTrailer(const std::string & libraryName);

      /// Split the library up into two words, around the capital letters
      typedef std::vector<std::pair<std::string, std::string> > DomainPackagePair;
      static DomainPackagePair
        twoWordsFrom(const std::string & libraryName);


    private:

      seal::PluginManager * thePluginManager;
      // maps each module to its library
      std::map<std::string, std::string> theLibraryMap;
    };

  }
}

#endif

