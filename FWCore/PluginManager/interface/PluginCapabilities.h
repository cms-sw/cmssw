#ifndef FWCore_PluginManager_PluginCapabilities_h
#define FWCore_PluginManager_PluginCapabilities_h
// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     PluginCapabilities
// 
/**\class PluginCapabilities PluginCapabilities.h FWCore/PluginManager/interface/PluginCapabilities.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Apr  6 12:36:19 EDT 2007
//

// system include files
#include <map>
#include <string>

// user include files
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
// forward declarations

namespace edmplugin {
  class SharedLibrary;
  class DummyFriend;
class PluginCapabilities : public PluginFactoryBase
{
   friend class DummyFriend;
   public:
      virtual ~PluginCapabilities();

      // ---------- const member functions ---------------------
      virtual std::vector<PluginInfo> available() const;
      virtual const std::string& category() const; 
      
      // ---------- static member functions --------------------
      static PluginCapabilities* get();
      
      // ---------- member functions ---------------------------
      void load(const std::string& iName);
      
      //returns false if loading fails because iName is unknown
      bool tryToLoad(const std::string& iName);
      
      ///Check to see if any capabilities are in the file, returns 'true' if found
      bool tryToFind(const SharedLibrary& iLoadable);

   private:
      PluginCapabilities();
      PluginCapabilities(const PluginCapabilities&); // stop default

      const PluginCapabilities& operator=(const PluginCapabilities&); // stop default

      // ---------- member data --------------------------------
      std::map<std::string, boost::filesystem::path> classToLoadable_;
};

}
#endif
