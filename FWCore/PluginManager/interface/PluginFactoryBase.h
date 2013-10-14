#ifndef FWCore_PluginManager_PluginFactoryBase_h
#define FWCore_PluginManager_PluginFactoryBase_h
// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     PluginFactoryBase
// 
/**\class PluginFactoryBase PluginFactoryBase.h FWCore/PluginManager/interface/PluginFactoryBase.h

 Description: Base class for all plugin factories

 Usage:
    This interface provides access to the most generic information about a plugin factory:
    1) what is the name of the presently available plugins
    2) the file name of the loadable which contained the plugin
*/
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 12:24:44 EDT 2007
//

// system include files
#include <string>
#include <vector>
#include <map>
#include <mutex>

#include "FWCore/Utilities/interface/Signal.h"
// user include files
#include "FWCore/PluginManager/interface/PluginInfo.h"

// forward declarations
namespace edmplugin {
class PluginFactoryBase
{

   public:
      PluginFactoryBase() {}
      virtual ~PluginFactoryBase();

      typedef std::vector<std::pair<void*, std::string> > PMakers;
      typedef std::map<std::string, PMakers > Plugins;

      // ---------- const member functions ---------------------

      ///return info about all plugins which are already available in the program
      virtual std::vector<PluginInfo> available() const;

      ///returns the name of the category to which this plugin factory belongs
      virtual const std::string& category() const = 0;
      
      ///signal containing plugin category, and  plugin info for newly added plugin
      mutable edm::signalslot::Signal<void(const std::string&, const PluginInfo&)> newPluginAdded_;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   protected:
      /**call this as the last line in the constructor of inheriting classes
      this is done so that the virtual table will be properly initialized when the
      routine is called
      */
      void finishedConstruction();
      
      void newPlugin(const std::string& iName);

      //since each inheriting class has its own Container type to hold their PMakers
      // this function allows them to share the same code when doing the lookup
      // this routine will throw an exception if iName is unknown therefore the return value is always valid
      void* findPMaker(const std::string& iName) const;

      //similar to findPMaker but will return 'end()' if iName is known
      void* tryToFindPMaker(const std::string& iName) const;
      
      void fillInfo(const PMakers &makers,
                    PluginInfo& iInfo,
                    std::vector<PluginInfo>& iReturn ) const;

      void fillAvailable(std::vector<PluginInfo>& iReturn) const;

      void registerPMaker(void* iPMaker, const std::string& iName);
      
   private:
      PluginFactoryBase(const PluginFactoryBase&); // stop default

      const PluginFactoryBase& operator=(const PluginFactoryBase&); // stop default

      void checkProperLoadable(const std::string& iName, const std::string& iLoadedFrom) const;
      // ---------- member data --------------------------------
      Plugins m_plugins;
      mutable std::recursive_mutex m_mutex;
  

};

}
#endif
