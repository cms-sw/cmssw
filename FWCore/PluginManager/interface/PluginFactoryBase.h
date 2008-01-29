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
// $Id: PluginFactoryBase.h,v 1.3 2007/04/27 12:09:42 chrjones Exp $
//

// system include files
#include <string>
#include <vector>
#include "sigc++/signal.h"
// user include files
#include "FWCore/PluginManager/interface/PluginInfo.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations
namespace edmplugin {
class PluginFactoryBase
{

   public:
      PluginFactoryBase() {}
      virtual ~PluginFactoryBase();

      // ---------- const member functions ---------------------

      ///return info about all plugins which are already available in the program
      virtual std::vector<PluginInfo> available() const = 0;

      ///returns the name of the category to which this plugin factory belongs
      virtual const std::string& category() const = 0;
      
      ///signal containing plugin category, and  plugin info for newly added plugin
      mutable sigc::signal<void,const std::string&, const PluginInfo&> newPluginAdded_;

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
      template<typename Plugins>
      typename Plugins::const_iterator findPMaker(const std::string& iName,
                                                   const Plugins& iPlugins) const {
        //do we already have it?
        typename Plugins::const_iterator itFound = iPlugins.find(iName);
        if(itFound == iPlugins.end()) {
          std::string lib = PluginManager::get()->load(this->category(),iName).path().native_file_string();
          itFound = iPlugins.find(iName);
          if(itFound == iPlugins.end()) {
            throw cms::Exception("PluginCacheError")<<"The plugin '"<<iName<<"' should have been in loadable\n '"
            <<lib<<"'\n but was not there.  This means the plugin cache is incorrect.  Please run 'EdmPluginRefresh "<<lib<<"'";
          }
        } else {
          //should check to see if this is from the proper loadable if it
          // was not statically linked
          if (itFound->second.front().second != PluginManager::staticallyLinkedLoadingFileName() &&
              PluginManager::isAvailable()) {
            if( itFound->second.front().second != PluginManager::get()->loadableFor(category(),iName).native_file_string() ) {
              throw cms::Exception("WrongPluginLoaded")<<"The plugin '"<<iName<<"' should have been loaded from\n '"
              <<PluginManager::get()->loadableFor(category(),iName).native_file_string()
              <<"'\n but instead it was already loaded from\n '"
              <<itFound->second.front().second<<"'\n because some other plugin was loaded from the latter loadables.\n"
              "To work around the problem the plugin '"<<iName<<"' should only be defined in one of these loadables.";
            }
          }
        }
        return itFound;
      }
      
      template<typename MakersItr>
        static void fillInfo(MakersItr iBegin, MakersItr iEnd,
                             PluginInfo& iInfo,
                             std::vector<PluginInfo>& iReturn ) {
          for(MakersItr it = iBegin;
              it != iEnd;
              ++it) {
            iInfo.loadable_ = it->second;
            iReturn.push_back(iInfo);
          }
        }
      template<typename PluginsItr>
      static void fillAvailable(PluginsItr iBegin,
                                PluginsItr iEnd,
                                std::vector<PluginInfo>& iReturn) {
        PluginInfo info;
        for( PluginsItr it = iBegin;
            it != iEnd;
            ++it) {
          info.name_ = it->first;
          fillInfo(it->second.begin(),it->second.end(),
                   info, iReturn);
        }
      }
      
      
   private:
      PluginFactoryBase(const PluginFactoryBase&); // stop default

      const PluginFactoryBase& operator=(const PluginFactoryBase&); // stop default

      // ---------- member data --------------------------------

};

}
#endif
