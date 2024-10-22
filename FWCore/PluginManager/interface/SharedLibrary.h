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
#include <filesystem>

// user include files

// forward declarations

namespace edmplugin {
  class SharedLibrary {
  public:
    SharedLibrary(const std::filesystem::path& iName);
    SharedLibrary(const SharedLibrary&) = delete;                   // stop default
    const SharedLibrary& operator=(const SharedLibrary&) = delete;  // stop default
    ~SharedLibrary();

    // ---------- const member functions ---------------------
    bool symbol(const std::string& iSymbolName, void*& iSymbol) const;
    const std::filesystem::path& path() const { return path_; }

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------

  private:
    // ---------- member data --------------------------------
    void* libraryHandle_;
    std::filesystem::path path_;
  };

}  // namespace edmplugin
#endif
