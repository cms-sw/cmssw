#include "TFileAdaptor.h"

#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"

#include <TROOT.h>
#include <TFile.h>
#include <TPluginManager.h>

#include <boost/shared_ptr.hpp>

#include <algorithm>
#include <sstream>

// Driver for configuring ROOT plug-in manager to use TStorageFactoryFile.
  void
  TFileAdaptor::addType(TPluginManager* mgr, char const* type) {
    mgr->AddHandler("TFile",
                    type,
                    "TStorageFactoryFile",
                    "IOPoolTFileAdaptor",
                    "TStorageFactoryFile(char const*,Option_t*,char const*,Int_t)");

    mgr->AddHandler("TSystem",
                    type,
                    "TStorageFactorySystem",
                    "IOPoolTFileAdaptor",
                    "TStorageFactorySystem()");
  }

  bool
  TFileAdaptor::native(char const* proto) const {
    return std::find(native_.begin(), native_.end(), "all") != native_.end()
      || std::find(native_.begin(), native_.end(), proto) != native_.end();
  }

  TFileAdaptor::TFileAdaptor(edm::ParameterSet const& p, edm::ActivityRegistry& ar)
    : enabled_(true),
      doStats_(true),
      cacheHint_("application-only"),
      readHint_("auto-detect"),
      tempDir_(),
      minFree_(0),
      timeout_(0U),
      native_() {
    if (!(enabled_ = p.getUntrackedParameter<bool> ("enable", enabled_)))
      return;

    StorageFactory* f = StorageFactory::get();
    doStats_ = p.getUntrackedParameter<bool> ("stats", doStats_);

    // values set in the site local config or in SiteLocalConfigService override
    // any values set here for this service.
    // These parameters here are needed only for backward compatibility
    // for WMDM tools until we switch to only using the site local config for this info.
    cacheHint_ = p.getUntrackedParameter<std::string> ("cacheHint", cacheHint_);
    readHint_ = p.getUntrackedParameter<std::string> ("readHint", readHint_);
    tempDir_ = p.getUntrackedParameter<std::string> ("tempDir", f->tempPath());
    minFree_ = p.getUntrackedParameter<double> ("tempMinFree", f->tempMinFree());
    native_ = p.getUntrackedParameter<std::vector<std::string> >("native", native_);

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
    }

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
      throw cms::Exception("TFileAdaptor")
        << "Unrecognised 'cacheHint' value '" << cacheHint_
        << "', recognised values are 'application-only',"
        << " 'storage-only', 'lazy-download', 'auto-detect'";

    if (readHint_ == "direct-unbuffered")
      f->setReadHint(StorageFactory::READ_HINT_UNBUFFERED);
    else if (readHint_ == "read-ahead-buffered")
      f->setReadHint(StorageFactory::READ_HINT_READAHEAD);
    else if (readHint_ == "auto-detect")
      f->setReadHint(StorageFactory::READ_HINT_AUTO);
    else
      throw cms::Exception("TFileAdaptor")
        << "Unrecognised 'readHint' value '" << readHint_
        << "', recognised values are 'direct-unbuffered',"
        << " 'read-ahead-buffered', 'auto-detect'";

    f->setTimeout(timeout_);

    // enable file access stats accounting if requested
    f->enableAccounting(doStats_);

    // tell where to save files.
    f->setTempDir(tempDir_, minFree_);

    // set our own root plugins
    TPluginManager* mgr = gROOT->GetPluginManager();
    mgr->LoadHandlersFromPluginDirs();

    if (!native("file"))      addType(mgr, "^file:");
    if (!native("http"))      addType(mgr, "^http:");
    if (!native("ftp"))       addType(mgr, "^ftp:");
    /* always */              addType(mgr, "^web:");
    /* always */              addType(mgr, "^gsiftp:");
    /* always */              addType(mgr, "^sfn:");
    if (!native("rfio"))      addType(mgr, "^rfio:");
    if (!native("dcache"))    addType(mgr, "^dcache:");
    if (!native("dcap"))      addType(mgr, "^dcap:");
    if (!native("gsidcap"))   addType(mgr, "^gsidcap:");
    if (!native("storm"))     addType(mgr, "^storm:");
    if (!native("storm-lcg")) addType(mgr, "^storm-lcg:");
  }

  void
  TFileAdaptor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addOptionalUntracked<bool>("enable");
    desc.addOptionalUntracked<bool>("stats");
    desc.addOptionalUntracked<std::string>("cacheHint");
    desc.addOptionalUntracked<std::string>("readHint");
    desc.addOptionalUntracked<std::string>("tempDir");
    desc.addOptionalUntracked<double>("tempMinFree");
    desc.addOptionalUntracked<std::vector<std::string> >("native");
    descriptions.add("AdaptorConfig", desc);
  }

  // Write current Storage statistics on a ostream
  void
  TFileAdaptor::termination(void) const {
    std::map<std::string, std::string> data;
    statsXML(data);
    if (!data.empty()) {
      edm::Service<edm::JobReport> reportSvc;
      reportSvc->reportPerformanceSummary("StorageStatistics", data);
    }
  }

  void
  TFileAdaptor::stats(std::ostream& o) const {
    if (!doStats_) {
      return;
    }
    float const oneMeg = 1048576.0;
    o << "Storage parameters: adaptor: true"
      << " Stats:" << (doStats_ ? "true" : "false") << '\n'
      << " Cache hint:" << cacheHint_ << '\n'
      << " Read hint:" << readHint_ << '\n'
      << "Storage statistics: "
      << StorageAccount::summaryText()
      << "; tfile/read=?/?/" << (TFile::GetFileBytesRead() / oneMeg) << "MB/?ms/?ms/?ms"
      << "; tfile/write=?/?/" << (TFile::GetFileBytesWritten() / oneMeg) << "MB/?ms/?ms/?ms";
  }

  void
  TFileAdaptor::statsXML(std::map<std::string, std::string>& data) const {
    if (!doStats_) {
      return;
    }
    float const oneMeg = 1048576.0;
    data.insert(std::make_pair("Parameter-untracked-bool-enabled", "true"));
    data.insert(std::make_pair("Parameter-untracked-bool-stats", (doStats_ ? "true" : "false")));
    data.insert(std::make_pair("Parameter-untracked-string-cacheHint", cacheHint_));
    data.insert(std::make_pair("Parameter-untracked-string-readHint", readHint_));
    StorageAccount::fillSummary(data);
    std::ostringstream r;
    std::ostringstream w;
    r << (TFile::GetFileBytesRead() / oneMeg);
    w << (TFile::GetFileBytesWritten() / oneMeg);
    data.insert(std::make_pair("ROOT-tfile-read-totalMegabytes", r.str()));
    data.insert(std::make_pair("ROOT-tfile-write-totalMegabytes", w.str()));
  }

/*
 * wrapper to bind TFileAdaptor to root, python etc
 * loading IOPoolTFileAdaptor library and instantiating
 * TFileAdaptorUI will make root to use StorageAdaptor for I/O instead
 * of its own plugins
 */
class TFileAdaptorUI {
public:

  TFileAdaptorUI();
  ~TFileAdaptorUI();

  // print current Storage statistics on cout
  void stats() const;

private:
  boost::shared_ptr<TFileAdaptor> me;
};

#include <iostream>

TFileAdaptorUI::TFileAdaptorUI() {
  edm::ActivityRegistry ar;
  const edm::ParameterSet param;
  me.reset(new TFileAdaptor(param, ar));
}

TFileAdaptorUI::~TFileAdaptorUI() {}

void TFileAdaptorUI::stats() const {
  me->stats(std::cout); std::cout << std::endl;
}

typedef TFileAdaptor AdaptorConfig;

DEFINE_FWK_SERVICE(AdaptorConfig);
