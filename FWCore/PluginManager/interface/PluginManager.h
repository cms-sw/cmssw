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
//

// system include files
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tbb/concurrent_unordered_map.h"

// user include files
#include "FWCore/Utilities/interface/Signal.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "FWCore/PluginManager/interface/PluginInfo.h"

// forward declarations
namespace edmplugin {
  class DummyFriend;
  class PluginFactoryBase;

  struct PluginManagerPathHasher {
    size_t operator()(std::filesystem::path const& iPath) const { return std::hash<std::string>{}(iPath.native()); }
  };

  class PluginManager {
    friend class DummyFriend;

  public:
    typedef std::vector<std::string> SearchPath;
    typedef std::vector<PluginInfo> Infos;
    typedef std::map<std::string, Infos> CategoryToInfos;

    class Config {
    public:
      Config() {}
      Config& searchPath(const SearchPath& iPath) {
        m_path = iPath;
        return *this;
      }
      const SearchPath& searchPath() const { return m_path; }
      void allowNoCache() { m_mustHaveCache = false; }

      bool mustHaveCache() const { return m_mustHaveCache; }

    private:
      SearchPath m_path;
      bool m_mustHaveCache = true;
    };

    PluginManager(const PluginManager&) = delete;                   // stop default
    const PluginManager& operator=(const PluginManager&) = delete;  // stop default
    ~PluginManager();

    // ---------- const member functions ---------------------
    const SharedLibrary& load(const std::string& iCategory, const std::string& iPlugin);

    const std::filesystem::path& loadableFor(const std::string& iCategory, const std::string& iPlugin);

    /**The container is ordered by category, then plugin name and then by precidence order of the plugin files.
        Therefore the first match on category and plugin name will be the proper file to load
        */
    const CategoryToInfos& categoryToInfos() const { return categoryToInfos_; }

    //If can not find iPlugin in category iCategory return null pointer, any other failure will cause a throw
    const SharedLibrary* tryToLoad(const std::string& iCategory, const std::string& iPlugin);

    // ---------- static member functions --------------------
    ///file name of the shared object being loaded
    static const std::string& loadingFile() { return loadingLibraryNamed_(); }

    ///if the value returned from loadingFile matches this string then the file is statically linked
    static const std::string& staticallyLinkedLoadingFileName();

    static PluginManager* get();
    static PluginManager& configure(const Config&);

    static bool isAvailable();

    // ---------- member functions ---------------------------
    edm::signalslot::Signal<void(const std::filesystem::path&)> goingToLoad_;
    edm::signalslot::Signal<void(const SharedLibrary&)> justLoaded_;
    edm::signalslot::Signal<void(const std::string&, const std::string&)> askedToLoadCategoryWithPlugin_;

  private:
    PluginManager(const Config&);

    void newFactory(const PluginFactoryBase*);
    static std::string& loadingLibraryNamed_();
    static PluginManager*& singleton();

    std::recursive_mutex& pluginLoadMutex() { return pluginLoadMutex_; }

    const std::filesystem::path& loadableFor_(const std::string& iCategory,
                                              const std::string& iPlugin,
                                              bool& ioThrowIfFailElseSucceedStatus);
    // ---------- member data --------------------------------
    SearchPath searchPath_;
    tbb::concurrent_unordered_map<std::filesystem::path, std::shared_ptr<SharedLibrary>, PluginManagerPathHasher>
        loadables_;

    CategoryToInfos categoryToInfos_;
    std::recursive_mutex pluginLoadMutex_;
  };

}  // namespace edmplugin
#endif
