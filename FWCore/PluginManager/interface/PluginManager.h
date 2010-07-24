#ifndef FWCore_PluginManager_PluginManager_h
#define FWCore_PluginManager_PluginManager_h
// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     PluginManager
// 
/**\class PluginManager PluginManager.h FWCore/PluginManager/interface/PluginManager.h

 Description: Manages the loading of shared objects

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 14:28:48 EDT 2007
// $Id: PluginManager.h,v 1.5 2007/07/03 19:16:33 chrjones Exp $
//

// system include files
#include <vector>
#include <map>
#include <string>
#include <boost/filesystem/path.hpp>
#include "sigc++/signal.h"

// user include files
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/PluginManager/interface/PluginInfo.h"

// forward declarations
namespace edmplugin {
  class DummyFriend;
  class PluginFactoryBase;
class PluginManager
{
   friend class DummyFriend;
  public:
     typedef std::vector<std::string> SearchPath;
     typedef std::vector<PluginInfo> Infos;
     typedef std::map<std::string, Infos > CategoryToInfos;

     class Config {
      public:
       Config() { }
       Config& searchPath(const SearchPath& iPath) {
         m_path = iPath;
         return *this;
       }
       const SearchPath& searchPath() const {
         return m_path;
       }
       private:
       SearchPath m_path;
     };

      ~PluginManager();

      // ---------- const member functions ---------------------
      const SharedLibrary& load(const std::string& iCategory,
                                const std::string& iPlugin);

      const boost::filesystem::path& loadableFor(const std::string& iCategory,
                                                 const std::string& iPlugin);
      
      /**The container is ordered by category, then plugin name and then by precidence order of the plugin files.
        Therefore the first match on category and plugin name will be the proper file to load
        */
      const CategoryToInfos& categoryToInfos() const {
        return categoryToInfos_;
      }
      
      //If can not find iPlugin in category iCategory return null pointer, any other failure will cause a throw
      const SharedLibrary* tryToLoad(const std::string& iCategory,
                                     const std::string& iPlugin);
      
      // ---------- static member functions --------------------
      ///file name of the shared object being loaded
      static const std::string& loadingFile() { 
        return loadingLibraryNamed_();}

      ///if the value returned from loadingFile matches this string then the file is statically linked
      static const std::string& staticallyLinkedLoadingFileName();

      
      static PluginManager* get();
      static PluginManager& configure(const Config& );
      
      static bool isAvailable();
      
      // ---------- member functions ---------------------------
      sigc::signal<void,const boost::filesystem::path&> goingToLoad_;
      sigc::signal<void,const SharedLibrary&> justLoaded_;
      sigc::signal<void,const std::string&,const std::string&> askedToLoadCategoryWithPlugin_;
   private:
      PluginManager(const Config&);
      PluginManager(const PluginManager&); // stop default

      const PluginManager& operator=(const PluginManager&); // stop default

      void newFactory(const PluginFactoryBase* );
      static std::string& loadingLibraryNamed_();
      static PluginManager*& singleton();
      
      const boost::filesystem::path& loadableFor_(const std::string& iCategory,
                                                  const std::string& iPlugin,
                                                  bool& ioThrowIfFailElseSucceedStatus);
      // ---------- member data --------------------------------
      SearchPath searchPath_;
      std::map<boost::filesystem::path, boost::shared_ptr<SharedLibrary> > loadables_;
      
      CategoryToInfos categoryToInfos_;
};

}
#endif
