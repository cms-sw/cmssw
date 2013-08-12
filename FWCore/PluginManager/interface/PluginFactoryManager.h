#ifndef FWCore_PluginManager_PluginFactoryManager_h
#define FWCore_PluginManager_PluginFactoryManager_h
// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     PluginFactoryManager
// 
/**\class PluginFactoryManager PluginFactoryManager.h FWCore/PluginManager/interface/PluginFactoryManager.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 12:49:41 EDT 2007
//

// system include files
#include <string>
#include <vector>
#include "FWCore/Utilities/interface/Signal.h"

// user include files

// forward declarations
namespace edmplugin {
  class PluginFactoryBase;
  class DummyFriend;
class PluginFactoryManager
{

   public:
      friend class DummyFriend;
  
      ~PluginFactoryManager();

      typedef std::vector<const PluginFactoryBase*>::const_iterator const_iterator;
      // ---------- const member functions ---------------------
      const_iterator begin() const;
      const_iterator end() const;

      // ---------- static member functions --------------------
      static PluginFactoryManager* get();

      // ---------- member functions ---------------------------
      void addFactory(const PluginFactoryBase*);
      edm::signalslot::Signal<void(const PluginFactoryBase*)> newFactory_;
      
   private:
      PluginFactoryManager();
      PluginFactoryManager(const PluginFactoryManager&); // stop default

      const PluginFactoryManager& operator=(const PluginFactoryManager&); // stop default

      // ---------- member data --------------------------------
      std::vector<const PluginFactoryBase*> factories_;

};

}
#endif
