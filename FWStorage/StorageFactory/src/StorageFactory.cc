#include <memory>

#include "FWStorage/StorageFactory/interface/StorageFactory.h"
#include "FWStorage/StorageFactory/interface/StorageMakerFactory.h"
#include "FWStorage/StorageFactory/interface/StorageAccount.h"
#include "FWStorage/StorageFactory/interface/StorageAccountProxy.h"
#include "FWStorage/StorageFactory/interface/StorageProxyMaker.h"
#include "FWStorage/StorageFactory/interface/LocalCacheFile.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace edm::storage;

StorageFactory StorageFactory::s_instance;

StorageFactory::StorageFactory(void)
    : m_cacheHint(CACHE_HINT_AUTO_DETECT),
      m_readHint(READ_HINT_AUTO),
      m_accounting(false),
      m_tempfree(defaultMinTempFree()),
      m_temppath(defaultTempDir()),
      m_timeout(0U),
      m_debugLevel(0U) {
  setTempDir(m_temppath, m_tempfree);
}

StorageFactory::~StorageFactory(void) {}

const StorageFactory *StorageFactory::get(void) { return &s_instance; }

StorageFactory *StorageFactory::getToModify(void) { return &s_instance; }

std::string StorageFactory::defaultTempDir() { return ".:$TMPDIR"; }

bool StorageFactory::enableAccounting(bool enabled) {
  bool old = m_accounting;
  m_accounting = enabled;
  return old;
}

bool StorageFactory::accounting(void) const { return m_accounting; }

void StorageFactory::setCacheHint(CacheHint value) { m_cacheHint = value; }

StorageFactory::CacheHint StorageFactory::cacheHint(void) const { return m_cacheHint; }

void StorageFactory::setReadHint(ReadHint value) { m_readHint = value; }

StorageFactory::ReadHint StorageFactory::readHint(void) const { return m_readHint; }

void StorageFactory::setTimeout(unsigned int timeout) { m_timeout = timeout; }

unsigned int StorageFactory::timeout(void) const { return m_timeout; }

void StorageFactory::setDebugLevel(unsigned int level) { m_debugLevel = level; }

unsigned int StorageFactory::debugLevel(void) const { return m_debugLevel; }

void StorageFactory::setTempDir(const std::string &s, double minFreeSpace) {
#if 0
  std::cerr /* edm::LogInfo("StorageFactory") */
    << "Considering path '" << s
    << "', min free space " << minFreeSpace
    << "GB for temp dir" << std::endl;
#endif

  size_t begin = 0;
  std::vector<std::string> dirs;
  dirs.reserve(std::count(s.begin(), s.end(), ':') + 1);

  while (true) {
    size_t end = s.find(':', begin);
    if (end == std::string::npos) {
      dirs.push_back(s.substr(begin, end));
      break;
    } else {
      dirs.push_back(s.substr(begin, end - begin));
      begin = end + 1;
    }
  }

  m_temppath = s;
  m_tempfree = minFreeSpace;
  std::tie(m_tempdir, m_unusableDirWarnings) = m_lfs.findCachePath(dirs, minFreeSpace);

#if 0
  std::cerr /* edm::LogInfo("StorageFactory") */
    << "Using '" << m_tempdir << "' for temp dir"
    << std::endl;
#endif
}

std::string StorageFactory::tempDir(void) const { return m_tempdir; }

std::string StorageFactory::tempPath(void) const { return m_temppath; }

double StorageFactory::tempMinFree(void) const { return m_tempfree; }

void StorageFactory::setStorageProxyMakers(std::vector<std::unique_ptr<StorageProxyMaker>> makers) {
  m_storageProxyMakers_ = std::move(makers);
}

StorageMaker *StorageFactory::getMaker(const std::string &proto) const {
  auto itFound = m_makers.find(proto);
  if (itFound != m_makers.end()) {
    return itFound->second.get();
  }
  if (!edmplugin::PluginManager::isAvailable()) {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }
  std::shared_ptr<StorageMaker> instance{StorageMakerFactory::get()->tryToCreate(proto)};
  auto insertResult = m_makers.insert(MakerTable::value_type(proto, instance));
  //Can't use instance since it is possible that another thread beat
  // us to the insertion so the map contains a different instance.
  return insertResult.first->second.get();
}

StorageMaker *StorageFactory::getMaker(const std::string &url, std::string &protocol, std::string &rest) const {
  size_t p = url.find(':');
  if (p != std::string::npos) {
    protocol = url.substr(0, p);
    rest = url.substr(p + 1);
  } else {
    protocol = "file";
    rest = url;
  }

  return getMaker(protocol);
}

std::unique_ptr<Storage> StorageFactory::open(const std::string &url, const int mode /* = IOFlags::OpenRead */) const {
  std::string protocol;
  std::string rest;
  std::unique_ptr<Storage> ret;
  std::unique_ptr<StorageAccount::Stamp> stats;
  if (StorageMaker *maker = getMaker(url, protocol, rest)) {
    if (m_accounting) {
      auto token = StorageAccount::tokenForStorageClassName(protocol);
      stats = std::make_unique<StorageAccount::Stamp>(StorageAccount::counter(token, StorageAccount::Operation::open));
    }
    try {
      ret = maker->open(
          protocol, rest, mode, StorageMaker::AuxSettings{}.setDebugLevel(m_debugLevel).setTimeout(m_timeout));
      if (ret) {
        // Inject proxy wrappers at the lowest level, in the order
        // specified in the configuration
        for (auto const &proxyMaker : m_storageProxyMakers_) {
          ret = proxyMaker->wrap(url, std::move(ret));
        }

        // Wrap the storage to LocalCacheFile if storage backend is
        // not already using a local file, and lazy-download is requested
        if (auto const useLocalFile = maker->usesLocalFile(); useLocalFile != StorageMaker::UseLocalFile::kNo) {
          bool useCacheFile;
          std::string path;
          if (useLocalFile == StorageMaker::UseLocalFile::kCheckFromPath) {
            path = rest;
          }
          std::tie(ret, useCacheFile) = wrapNonLocalFile(std::move(ret), protocol, path, mode);
          if (useCacheFile) {
            protocol = "local-cache";
          }
        }

        if (m_accounting)
          ret = std::make_unique<StorageAccountProxy>(protocol, std::move(ret));

        if (stats)
          stats->tick();
      }
    } catch (cms::Exception &err) {
      err.addContext("Calling StorageFactory::open()");
      err.addAdditionalInfo(err.message());
      err.clearMessage();
      err << "Failed to open the file '" << url << "'";
      throw;
    }
  }
  return ret;
}

void StorageFactory::stagein(const std::string &url) const {
  std::string protocol;
  std::string rest;

  std::unique_ptr<StorageAccount::Stamp> stats;
  if (StorageMaker *maker = getMaker(url, protocol, rest)) {
    if (m_accounting) {
      auto token = StorageAccount::tokenForStorageClassName(protocol);
      stats =
          std::make_unique<StorageAccount::Stamp>(StorageAccount::counter(token, StorageAccount::Operation::stagein));
    }
    try {
      maker->stagein(protocol, rest, StorageMaker::AuxSettings{}.setDebugLevel(m_debugLevel).setTimeout(m_timeout));
      if (stats)
        stats->tick();
    } catch (cms::Exception &err) {
      edm::LogWarning("StorageFactory::stagein()") << "Failed to stage in file '" << url << "' because:\n"
                                                   << err.explainSelf();
    }
  }
}

bool StorageFactory::check(const std::string &url, IOOffset *size /* = 0 */) const {
  std::string protocol;
  std::string rest;

  bool ret = false;
  std::unique_ptr<StorageAccount::Stamp> stats;
  if (StorageMaker *maker = getMaker(url, protocol, rest)) {
    if (m_accounting) {
      auto token = StorageAccount::tokenForStorageClassName(protocol);
      stats = std::make_unique<StorageAccount::Stamp>(StorageAccount::counter(token, StorageAccount::Operation::check));
    }
    try {
      ret = maker->check(
          protocol, rest, StorageMaker::AuxSettings{}.setDebugLevel(m_debugLevel).setTimeout(m_timeout), size);
      if (stats)
        stats->tick();
    } catch (cms::Exception &err) {
      edm::LogWarning("StorageFactory::check()")
          << "Existence or size check for the file '" << url << "' failed because:\n"
          << err.explainSelf();
    }
  }

  return ret;
}

std::tuple<std::unique_ptr<Storage>, bool> StorageFactory::wrapNonLocalFile(std::unique_ptr<Storage> s,
                                                                            const std::string &proto,
                                                                            const std::string &path,
                                                                            const int mode) const {
  StorageFactory::CacheHint hint = cacheHint();
  bool useCacheFile = false;
  if ((hint == StorageFactory::CACHE_HINT_LAZY_DOWNLOAD) || (mode & IOFlags::OpenWrap)) {
    if (mode & IOFlags::OpenWrite) {
      // For now, issue no warning - otherwise, we'd always warn on output files.
    } else if (m_tempdir.empty()) {
      edm::LogWarning("StorageFactory") << m_unusableDirWarnings;
    } else if ((not path.empty()) and m_lfs.isLocalPath(path)) {
      // For now, issue no warning - otherwise, we'd always warn on local input files.
    } else {
      if (accounting()) {
        s = std::make_unique<StorageAccountProxy>(proto, std::move(s));
      }
      s = std::make_unique<LocalCacheFile>(std::move(s), m_tempdir);
      useCacheFile = true;
    }
  }

  return {std::move(s), useCacheFile};
}
