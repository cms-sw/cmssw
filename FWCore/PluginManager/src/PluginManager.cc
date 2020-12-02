// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     PluginManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 14:28:58 EDT 2007
//

// system include files
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>

// TEMPORARY
#include "TInterpreter.h"
#include "TVirtualMutex.h"

// user include files
#include "FWCore/PluginManager/interface/CacheParser.h"
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
#include "FWCore/PluginManager/interface/PluginFactoryManager.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace edmplugin {
  //
  // constants, enums and typedefs
  //

  //
  // static data member definitions
  //

  static bool readCacheFile(const std::filesystem::path& cacheFile,
                            const std::filesystem::path& dir,
                            PluginManager::CategoryToInfos& categoryToInfos) {
    if (exists(cacheFile)) {
      std::ifstream file(cacheFile.string().c_str());
      if (not file) {
        throw cms::Exception("PluginMangerCacheProblem")
            << "Unable to open the cache file '" << cacheFile.string() << "'. Please check permissions on file";
      }
      CacheParser::read(file, dir, categoryToInfos);
      return true;
    }
    return false;
  }
  //
  // constructors and destructor
  //
  PluginManager::PluginManager(const PluginManager::Config& iConfig) : searchPath_(iConfig.searchPath()) {
    using std::placeholders::_1;
    const std::filesystem::path& kCacheFile(standard::cachefileName());
    // This is the filename of a file which contains plugins which exist in the
    // base release and which should exists in the local area, otherwise they
    // were removed and we want to catch their usage.
    const std::filesystem::path& kPoisonedCacheFile(standard::poisonedCachefileName());
    //NOTE: This may not be needed :/
    PluginFactoryManager* pfm = PluginFactoryManager::get();
    pfm->newFactory_.connect(std::bind(std::mem_fn(&PluginManager::newFactory), this, _1));

    // When building a single big executable the plugins are already registered in the
    // PluginFactoryManager, we therefore only need to populate the categoryToInfos_ map
    // with the relevant information.
    for (PluginFactoryManager::const_iterator i = pfm->begin(), e = pfm->end(); i != e; ++i) {
      categoryToInfos_[(*i)->category()] = (*i)->available();
    }

    //read in the files
    //Since we are looping in the 'precidence' order then the lists in categoryToInfos_ will also be
    // in that order
    bool foundAtLeastOneCacheFile = false;
    std::set<std::string> alreadySeen;
    for (SearchPath::const_iterator itPath = searchPath_.begin(), itEnd = searchPath_.end(); itPath != itEnd;
         ++itPath) {
      //take care of the case where the same path is passed in multiple times
      if (alreadySeen.find(*itPath) != alreadySeen.end()) {
        continue;
      }
      alreadySeen.insert(*itPath);
      std::filesystem::path dir(*itPath);
      if (exists(dir)) {
        if (not is_directory(dir)) {
          throw cms::Exception("PluginManagerBadPath")
              << "The path '" << dir.string() << "' for the PluginManager is not a directory";
        }
        std::filesystem::path cacheFile = dir / kCacheFile;

        if (readCacheFile(cacheFile, dir, categoryToInfos_)) {
          foundAtLeastOneCacheFile = true;
        }

        // We do not check for return code since we do not want to consider a
        // poison cache file as a valid cache file having been found.
        std::filesystem::path poisonedCacheFile = dir / kPoisonedCacheFile;
        readCacheFile(poisonedCacheFile, dir / "poisoned", categoryToInfos_);
      }
    }
    if (not foundAtLeastOneCacheFile and iConfig.mustHaveCache()) {
      auto ex = cms::Exception("PluginManagerNoCacheFile")
                << "No cache files named '" << standard::cachefileName() << "' were found in the directories \n";
      for (auto const& seen : alreadySeen) {
        ex << " '" << seen << "'\n";
      }
      throw ex;
    }
    //Since this should not be called until after 'main' has started, we can set the value
    loadingLibraryNamed_() = "<loaded by another plugin system>";
  }

  // PluginManager::PluginManager(const PluginManager& rhs)
  // {
  //    // do actual copying here;
  // }

  PluginManager::~PluginManager() {}

  //
  // assignment operators
  //
  // const PluginManager& PluginManager::operator=(const PluginManager& rhs)
  // {
  //   //An exception safe implementation is
  //   PluginManager temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }

  //
  // member functions
  //
  void PluginManager::newFactory(const PluginFactoryBase*) {}
  //
  // const member functions
  //
  namespace {
    struct PICompare {
      bool operator()(const PluginInfo& iLHS, const PluginInfo& iRHS) const { return iLHS.name_ < iRHS.name_; }
    };
  }  // namespace

  const std::filesystem::path& PluginManager::loadableFor(const std::string& iCategory, const std::string& iPlugin) {
    bool throwIfFail = true;
    return loadableFor_(iCategory, iPlugin, throwIfFail);
  }

  const std::filesystem::path& PluginManager::loadableFor_(const std::string& iCategory,
                                                           const std::string& iPlugin,
                                                           bool& ioThrowIfFailElseSucceedStatus) {
    const bool throwIfFail = ioThrowIfFailElseSucceedStatus;
    ioThrowIfFailElseSucceedStatus = true;
    CategoryToInfos::iterator itFound = categoryToInfos_.find(iCategory);
    if (itFound == categoryToInfos_.end()) {
      if (throwIfFail) {
        throw cms::Exception("PluginNotFound") << "Unable to find plugin '" << iPlugin << "' because the category '"
                                               << iCategory << "' has no known plugins";
      } else {
        ioThrowIfFailElseSucceedStatus = false;
        static const std::filesystem::path s_path;
        return s_path;
      }
    }

    PluginInfo i;
    i.name_ = iPlugin;
    typedef std::vector<PluginInfo>::iterator PIItr;
    std::pair<PIItr, PIItr> range = std::equal_range(itFound->second.begin(), itFound->second.end(), i, PICompare());

    if (range.first == range.second) {
      if (throwIfFail) {
        throw cms::Exception("PluginNotFound") << "Unable to find plugin '" << iPlugin << "' in category '" << iCategory
                                               << "'. Please check spelling of name.";
      } else {
        ioThrowIfFailElseSucceedStatus = false;
        static const std::filesystem::path s_path;
        return s_path;
      }
    }

    if (range.second - range.first > 1) {
      //see if the come from the same directory
      if (range.first->loadable_.parent_path() == (range.first + 1)->loadable_.parent_path()) {
        //std::cout<<range.first->name_ <<" " <<(range.first+1)->name_<<std::endl;
        throw cms::Exception("MultiplePlugins")
            << "The plugin '" << iPlugin
            << "' is found in multiple files \n"
               " '"
            << range.first->loadable_.filename() << "'\n '" << (range.first + 1)->loadable_.filename()
            << "'\n"
               "in directory '"
            << range.first->loadable_.parent_path().string()
            << "'.\n"
               "The code must be changed so the plugin only appears in one plugin file. "
               "You will need to remove the macro which registers the plugin so it only appears in"
               " one of these files.\n"
               "  If none of these files register such a plugin, "
               "then the problem originates in a library to which all these files link.\n"
               "The plugin registration must be removed from that library since plugins are not allowed in regular "
               "libraries.";
      }
    }

    return range.first->loadable_;
  }

  namespace {
    class Sentry {
    public:
      Sentry(std::string& iPath, const std::string& iNewPath) : path_(iPath), oldPath_(iPath) { path_ = iNewPath; }
      ~Sentry() { path_ = oldPath_; }

    private:
      std::string& path_;
      std::string oldPath_;
    };
  }  // namespace

  const SharedLibrary& PluginManager::load(const std::string& iCategory, const std::string& iPlugin) {
    askedToLoadCategoryWithPlugin_(iCategory, iPlugin);
    const std::filesystem::path& p = loadableFor(iCategory, iPlugin);

    //have we already loaded this?
    auto itLoaded = loadables_.find(p);
    if (itLoaded == loadables_.end()) {
      //Need to make sure we only have on SharedLibrary loading at a time
      std::lock_guard<std::recursive_mutex> guard(pluginLoadMutex());
      //Another thread may have gotten this while we were waiting on the mutex
      itLoaded = loadables_.find(p);
      if (itLoaded == loadables_.end()) {
        //try to make one
        goingToLoad_(p);
        Sentry s(loadingLibraryNamed_(), p.string());
        //std::filesystem::path native(p.string());
        std::shared_ptr<SharedLibrary> ptr;
        {
          //TEMPORARY: to avoid possible deadlocks from ROOT, we must
          // take the lock ourselves
          R__LOCKGUARD2(gInterpreterMutex);
          try {
            ptr = std::make_shared<SharedLibrary>(p);
          } catch (cms::Exception& e) {
            e.addContext("While attempting to load plugin " + iPlugin);
            throw;
          }
        }
        loadables_.emplace(p, ptr);
        justLoaded_(*ptr);
        return *ptr;
      }
    }
    return *(itLoaded->second);
  }

  const SharedLibrary* PluginManager::tryToLoad(const std::string& iCategory, const std::string& iPlugin) {
    askedToLoadCategoryWithPlugin_(iCategory, iPlugin);
    bool ioThrowIfFailElseSucceedStatus = false;
    const std::filesystem::path& p = loadableFor_(iCategory, iPlugin, ioThrowIfFailElseSucceedStatus);

    if (not ioThrowIfFailElseSucceedStatus) {
      return nullptr;
    }

    //have we already loaded this?
    auto itLoaded = loadables_.find(p);
    if (itLoaded == loadables_.end()) {
      //Need to make sure we only have on SharedLibrary loading at a time
      std::lock_guard<std::recursive_mutex> guard(pluginLoadMutex());
      //Another thread may have gotten this while we were waiting on the mutex
      itLoaded = loadables_.find(p);
      if (itLoaded == loadables_.end()) {
        //try to make one
        goingToLoad_(p);
        Sentry s(loadingLibraryNamed_(), p.string());
        //std::filesystem::path native(p.string());
        std::shared_ptr<SharedLibrary> ptr;
        {
          //TEMPORARY: to avoid possible deadlocks from ROOT, we must
          // take the lock ourselves
          R__LOCKGUARD(gInterpreterMutex);
          try {
            ptr = std::make_shared<SharedLibrary>(p);
          } catch (cms::Exception& e) {
            e.addContext("While attempting to load plugin " + iPlugin);
            throw;
          }
        }
        loadables_[p] = ptr;
        justLoaded_(*ptr);
        return ptr.get();
      }
    }
    return (itLoaded->second).get();
  }

  //
  // static member functions
  //
  PluginManager* PluginManager::get() {
    PluginManager* manager = singleton();
    if (nullptr == manager) {
      throw cms::Exception("PluginManagerNotConfigured")
          << "PluginManager::get() was called before PluginManager::configure.";
    }
    return manager;
  }

  PluginManager& PluginManager::configure(const Config& iConfig) {
    PluginManager*& s = singleton();
    if (nullptr != s) {
      throw cms::Exception("PluginManagerReconfigured");
    }

    const Config& realConfig = iConfig;
    if (realConfig.searchPath().empty()) {
      throw cms::Exception("PluginManagerEmptySearchPath");
    }
    s = new PluginManager(realConfig);
    return *s;
  }

  const std::string& PluginManager::staticallyLinkedLoadingFileName() {
    static const std::string s_name("static");
    return s_name;
  }

  std::string& PluginManager::loadingLibraryNamed_() {
    //NOTE: pluginLoadMutex() indirectly guards this since this value
    // is only accessible via the Sentry call which us guarded by the mutex
    CMS_THREAD_SAFE static std::string s_name(staticallyLinkedLoadingFileName());
    return s_name;
  }

  PluginManager*& PluginManager::singleton() {
    CMS_THREAD_SAFE static PluginManager* s_singleton = nullptr;
    return s_singleton;
  }

  bool PluginManager::isAvailable() { return nullptr != singleton(); }

}  // namespace edmplugin
