#ifndef FWCore_PluginManager_SharedLibrary_h
#define FWCore_PluginManager_SharedLibrary_h
// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     SharedLibrary
// 
/**\class SharedLibrary SharedLibrary.h FWCore/PluginManager/interface/SharedLibrary.h

 Description: Handles the loading of a SharedLibrary

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Apr  5 15:30:08 EDT 2007
//

// system include files
#include <boost/filesystem/path.hpp>

// user include files

// forward declarations

namespace edmplugin {
class SharedLibrary
{

   public:
      SharedLibrary(const boost::filesystem::path& iName);
      ~SharedLibrary();

      // ---------- const member functions ---------------------
      bool symbol(const std::string& iSymbolName, void* & iSymbol) const;
      const boost::filesystem::path& path() const { return path_;}

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      
   private:
      SharedLibrary(const SharedLibrary&) = delete; // stop default

      const SharedLibrary& operator=(const SharedLibrary&) = delete; // stop default

      // ---------- member data --------------------------------
      void* libraryHandle_;
      boost::filesystem::path path_;
};

}
#endif
