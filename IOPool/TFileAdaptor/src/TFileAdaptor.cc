#include "TFileAdaptor.h"

#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"
#include "FWCore/Reflection/interface/SetClassParsing.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageProxyMaker.h"
#include "Utilities/StorageFactory/interface/StorageProxyMakerFactory.h"

#include <TROOT.h>
#include <TFile.h>
#include <TPluginManager.h>

#include <memory>

#include <algorithm>
#include <sstream>

// Driver for configuring ROOT plug-in manager to use TStorageFactoryFile.

/**
   Register TFileAdaptor to be the handler for a given type.

   Once registered, URLs matching a specified regexp (for example, ^root: to
   manage files starting with root://) will be managed by a TFileAdaptor instance,
   possibly overriding any built-in ROOT adaptors.

   @param[in] mgr      The ROOT plugin manager object.
   @param[in] type     A regexp-string; URLs matching this string will use TFileAdaptor.
   @param[in] altType  Due to a limitation in the TPluginManager, if the type was 
                       previously managed by TXNetFile, we must invoke AddHandler with
                       a slightly different syntax.  Set this parameter to 1 if this
                       applies to you.  Otherwise, leave it at the default (0)
 */
void TFileAdaptor::addType(TPluginManager* mgr, char const* type, int altType /*=0*/) {
  // HACK:
  // The ROOT plug-in manager does not understand loading plugins with different
  // signatures.  So, because TXNetSystem is registered with a different constructor
  // than all the other plugins, we must match its interface in order to override
  // it.
  if (altType == 0) {
    mgr->AddHandler("TFile",
                    type,
                    "TStorageFactoryFile",
                    "IOPoolTFileAdaptor",
                    "TStorageFactoryFile(char const*,Option_t*,char const*,Int_t)");

    mgr->AddHandler("TSystem", type, "TStorageFactorySystem", "IOPoolTFileAdaptor", "TStorageFactorySystem()");
  } else if (altType == 1) {
    mgr->AddHandler("TFile",
                    type,
                    "TStorageFactoryFile",
                    "IOPoolTFileAdaptor",
                    "TStorageFactoryFile(char const*,Option_t*,char const*,Int_t, Int_t, Bool_t)");

    mgr->AddHandler(
        "TSystem", type, "TStorageFactorySystem", "IOPoolTFileAdaptor", "TStorageFactorySystem(const char *,Bool_t)");
  }
}

bool TFileAdaptor::native(char const* proto) const {
  return std::find(native_.begin(), native_.end(), "all") != native_.end() ||
         std::find(native_.begin(), native_.end(), proto) != native_.end();
}

TFileAdaptor::TFileAdaptor(edm::ParameterSet const& pset, edm::ActivityRegistry& ar)
    : enabled_(pset.getUntrackedParameter<bool>("enable")),
      doStats_(pset.getUntrackedParameter<bool>("stats")),
      enablePrefetching_(false),
      // values set in the site local config or in SiteLocalConfigService override
      // any values set here for this service.
      // These parameters here are needed only for backward compatibility
      // for WMDM tools until we switch to only using the site local config for this info.
      cacheHint_(pset.getUntrackedParameter<std::string>("cacheHint")),
      readHint_(pset.getUntrackedParameter<std::string>("readHint")),
      tempDir_(pset.getUntrackedParameter<std::string>("tempDir")),
      minFree_(pset.getUntrackedParameter<double>("tempMinFree")),
      native_(pset.getUntrackedParameter<std::vector<std::string>>("native")),
      // end of section of values overridden by SiteLocalConfigService
      timeout_(0U),
      debugLevel_(0U) {
  if (not enabled_)
    return;

  using namespace edm::storage;
  StorageFactory* f = StorageFactory::getToModify();

  ar.watchPostEndJob(this, &TFileAdaptor::termination);

  // Retrieve values from SiteLocalConfigService.
  // Any such values will override values set above.
  edm::Service<edm::SiteLocalConfig> pSLC;
  if (pSLC.isAvailable()) {
    if (std::string const* p = pSLC->sourceCacheTempDir()) {
      tempDir_ = *p;
    }
    if (double const* p = pSLC->sourceCacheMinFree()) {
      minFree_ = *p;
    }
    if (std::string const* p = pSLC->sourceCacheHint()) {
      cacheHint_ = *p;
    }
    if (std::string const* p = pSLC->sourceReadHint()) {
      readHint_ = *p;
    }
    if (unsigned int const* p = pSLC->sourceTimeout()) {
      timeout_ = *p;
    }
    if (std::vector<std::string> const* p = pSLC->sourceNativeProtocols()) {
      native_ = *p;
    }
    debugLevel_ = pSLC->debugLevel();
    enablePrefetching_ = pSLC->enablePrefetching();
  }

  // Prefetching does not work with storage-only; forcibly disable it.
  if ((enablePrefetching_) && ((cacheHint_ == "storage-only") || (cacheHint_ == "auto-detect")))
    cacheHint_ = "application-only";

  // tell factory how clients should access files
  if (cacheHint_ == "application-only")
    f->setCacheHint(StorageFactory::CACHE_HINT_APPLICATION);
  else if (cacheHint_ == "storage-only")
    f->setCacheHint(StorageFactory::CACHE_HINT_STORAGE);
  else if (cacheHint_ == "lazy-download")
    f->setCacheHint(StorageFactory::CACHE_HINT_LAZY_DOWNLOAD);
  else if (cacheHint_ == "auto-detect")
    f->setCacheHint(StorageFactory::CACHE_HINT_AUTO_DETECT);
  else
    throw cms::Exception("TFileAdaptor") << "Unrecognised 'cacheHint' value '" << cacheHint_
                                         << "', recognised values are 'application-only',"
                                         << " 'storage-only', 'lazy-download', 'auto-detect'";

  if (readHint_ == "direct-unbuffered")
    f->setReadHint(StorageFactory::READ_HINT_UNBUFFERED);
  else if (readHint_ == "read-ahead-buffered")
    f->setReadHint(StorageFactory::READ_HINT_READAHEAD);
  else if (readHint_ == "auto-detect")
    f->setReadHint(StorageFactory::READ_HINT_AUTO);
  else
    throw cms::Exception("TFileAdaptor") << "Unrecognised 'readHint' value '" << readHint_
                                         << "', recognised values are 'direct-unbuffered',"
                                         << " 'read-ahead-buffered', 'auto-detect'";

  f->setTimeout(timeout_);
  f->setDebugLevel(debugLevel_);

  // enable file access stats accounting if requested
  f->enableAccounting(doStats_);

  // tell where to save files.
  f->setTempDir(tempDir_, minFree_);

  // forward generic storage proxy makers
  {
    std::vector<std::unique_ptr<StorageProxyMaker>> makers;
    for (auto const& pset : pset.getUntrackedParameter<std::vector<edm::ParameterSet>>("storageProxies")) {
      makers.push_back(StorageProxyMakerFactory::get()->create(pset.getUntrackedParameter<std::string>("type"), pset));
    }
    f->setStorageProxyMakers(std::move(makers));
  }

  // set our own root plugins
  TPluginManager* mgr = gROOT->GetPluginManager();

  // Make sure ROOT parses system directories first.
  // Then our AddHandler() calls will also remove an existing handler
  // that was registered with the same regex
  mgr->LoadHandlersFromPluginDirs("TFile");
  mgr->LoadHandlersFromPluginDirs("TSystem");

  // Note: if you add a new handler, please update the test/tfileTest.cpp as well
  if (!native("file"))
    addType(mgr, "^file:");
  if (!native("http"))
    addType(mgr, "^http:");
  if (!native("http"))
    addType(mgr, "^http[s]?:");
  if (!native("ftp"))
    addType(mgr, "^ftp:");
  /* always */ addType(mgr, "^web:");
  if (!native("dcache"))
    addType(mgr, "^dcache:");
  if (!native("dcap"))
    addType(mgr, "^dcap:");
  if (!native("gsidcap"))
    addType(mgr, "^gsidcap:");
  if (!native("root"))
    addType(mgr, "^root:", 1);  // See comments in addType
  if (!native("root"))
    addType(mgr, "^[x]?root:", 1);  // See comments in addType

  // Make sure the TStorageFactoryFile can be loaded regardless of the header auto-parsing setting
  {
    edm::SetClassParsing guard(true);
    if (auto cl = TClass::GetClass("TStorageFactoryFile")) {
      cl->GetClassInfo();
    } else {
      throw cms::Exception("TFileAdaptor") << "Unable to obtain TClass for TStorageFactoryFile";
    }
  }
}

void TFileAdaptor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  using namespace edm::storage;
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("enable", true)->setComment("Enable or disable TFileAdaptor behavior");
  desc.addUntracked<bool>("stats", true);
  desc.addUntracked<std::string>("cacheHint", "auto-detect")
      ->setComment(
          "Hint for read caching. Possible values: 'application-only', 'storage-only', 'lazy-download', 'auto-detect'. "
          "The value from the SiteLocalConfigService overrides the value set here. In addition, if the "
          "SiteLocalConfigService has prefetching enabled, the default hint is 'application-only'.");
  desc.addUntracked<std::string>("readHint", "auto-detect")
      ->setComment(
          "Hint for reading itself. Possible values: 'direct-unbuffered', 'read-ahead-buffered', 'auto-detect'. The "
          "value from SiteLocalConfigService overrides the value set here.");
  desc.addUntracked<std::string>("tempDir", StorageFactory::defaultTempDir())
      ->setComment(
          "Colon-separated list of directories that storage implementations downloading the full file could place the "
          "file. The value from SiteLocalConfigService overrides the value set here.");
  desc.addUntracked<double>("tempMinFree", StorageFactory::defaultMinTempFree())
      ->setComment(
          "Minimum amount of space in GB required for a temporary data directory specified in tempDir. The value from "
          "SiteLocalConfigService overrides the value set here.");
  desc.addUntracked<std::vector<std::string>>("native", {})
      ->setComment(
          "Set of protocols for which to use a native ROOT storage implementation instead of CMSSW's StorageFactory. "
          "Valid "
          "values are 'file', 'http', 'ftp', 'dcache', 'dcap', 'gsidcap', 'root', or 'all' to prefer ROOT for all "
          "protocols. The value from SiteLocalConfigService overrides the value set here.");

  edm::ParameterSetDescription proxyMakerDesc;
  proxyMakerDesc.addNode(edm::PluginDescription<edm::storage::StorageProxyMakerFactory>("type", false));
  std::vector<edm::ParameterSet> proxyMakerDefaults;
  desc.addVPSetUntracked("storageProxies", proxyMakerDesc, proxyMakerDefaults)
      ->setComment(
          "Ordered list of Storage proxies the real Storage object is wrapped into. The real Storage is wrapped into "
          "the first element of the list, then that proxy is wrapped into the second element of the list and so on. "
          "Only after this wrapping are the LocalCacheFile (lazy-download) and statistics accounting ('stats' "
          "parameter) proxies applied.");

  descriptions.add("AdaptorConfig", desc);
  descriptions.setComment(
      "AdaptorConfig Service is used to configure the TFileAdaptor. If enabled, the TFileAdaptor registers "
      "TStorageFactoryFile as a handler for various protocols. The StorageFactory facility provides custom storage "
      "access implementations for these protocols, as well as statistics accounting.");
}

// Write current Storage statistics on a ostream
void TFileAdaptor::termination(void) const {
  std::map<std::string, std::string> data;
  statsXML(data);
  if (!data.empty()) {
    edm::Service<edm::JobReport> reportSvc;
    reportSvc->reportPerformanceSummary("StorageStatistics", data);
  }
}

void TFileAdaptor::stats(std::ostream& o) const {
  if (!doStats_) {
    return;
  }
  float const oneMeg = 1048576.0;
  o << "Storage parameters: adaptor: true"
    << " Stats:" << (doStats_ ? "true" : "false") << '\n'
    << " Prefetching:" << (enablePrefetching_ ? "true" : "false") << '\n'
    << " Cache hint:" << cacheHint_ << '\n'
    << " Read hint:" << readHint_ << '\n'
    << "Storage statistics: " << edm::storage::StorageAccount::summaryText() << "; tfile/read=?/?/"
    << (TFile::GetFileBytesRead() / oneMeg) << "MB/?ms/?ms/?ms"
    << "; tfile/write=?/?/" << (TFile::GetFileBytesWritten() / oneMeg) << "MB/?ms/?ms/?ms";
}

void TFileAdaptor::statsXML(std::map<std::string, std::string>& data) const {
  if (!doStats_) {
    return;
  }
  float const oneMeg = 1048576.0;
  data.insert(std::make_pair("Parameter-untracked-bool-enabled", "true"));
  data.insert(std::make_pair("Parameter-untracked-bool-stats", (doStats_ ? "true" : "false")));
  data.insert(std::make_pair("Parameter-untracked-bool-prefetching", (enablePrefetching_ ? "true" : "false")));
  data.insert(std::make_pair("Parameter-untracked-string-cacheHint", cacheHint_));
  data.insert(std::make_pair("Parameter-untracked-string-readHint", readHint_));
  edm::storage::StorageAccount::fillSummary(data);
  std::ostringstream r;
  std::ostringstream w;
  r << (TFile::GetFileBytesRead() / oneMeg);
  w << (TFile::GetFileBytesWritten() / oneMeg);
  data.insert(std::make_pair("ROOT-tfile-read-totalMegabytes", r.str()));
  data.insert(std::make_pair("ROOT-tfile-write-totalMegabytes", w.str()));
}

#include <iostream>

TFileAdaptorUI::TFileAdaptorUI() {
  edm::ActivityRegistry ar;
  const edm::ParameterSet param;
  me = std::make_shared<TFileAdaptor>(param, ar);  // propagate_const<T> has no reset() function
}

TFileAdaptorUI::~TFileAdaptorUI() {}

void TFileAdaptorUI::stats() const {
  me->stats(std::cout);
  std::cout << std::endl;
}
