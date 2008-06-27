#include "IOPool/TFileAdaptor/interface/TStorageFactoryFile.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <TROOT.h>
#include <TPluginManager.h>
#include <TFile.h>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

// Driver for configuring ROOT plug-in manager to use TStorageFactoryFile.
class TFileAdaptor
{
  bool enabled_;
  bool doStats_;
  std::string cacheHint_;
  std::string readHint_;
  std::string tempDir_;
  std::vector<std::string> native_;

  static void addType(TPluginManager *mgr, const char *type)
  {
    mgr->AddHandler("TFile", 
		    type, 
		    "TStorageFactoryFile", 
		    "IOPoolTFileAdaptor",
		    "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)"); 

    mgr->AddHandler("TSystem", 
		    type, 
		    "TStorageFactorySystem", 
		    "IOPoolTFileAdaptor",
		    "TStorageFactorySystem()"); 
  }

  bool native(const char *proto) const
  {
    return std::find(native_.begin(), native_.end(), "all") != native_.end()
      || std::find(native_.begin(), native_.end(), proto) != native_.end();
  }

public:
  TFileAdaptor(const edm::ParameterSet &p, edm::ActivityRegistry &ar)
    : enabled_(true),
      doStats_(false),
      cacheHint_("application-only"),
      readHint_("auto-detect"),
      tempDir_(".")
  {
    if (! (enabled_ = p.getUntrackedParameter<bool> ("enable", enabled_)))
      return;

    StorageFactory *f = StorageFactory::get();
    doStats_ = p.getUntrackedParameter<bool> ("stats", doStats_);
    cacheHint_ = p.getUntrackedParameter<std::string> ("cacheHint", cacheHint_);
    readHint_ = p.getUntrackedParameter<std::string> ("readHint", readHint_);
    tempDir_ = p.getUntrackedParameter<std::string> ("tempDir", tempDir_);
    native_ = p.getUntrackedParameter<std::vector<std::string> >("native", native_);
    ar.watchPostEndJob(this, &TFileAdaptor::termination);

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

    // enable file access stats accounting if requested
    f->enableAccounting(doStats_);

    // tell where to save files.
    f->setTempDir(tempDir_);

    // set our own root plugins
    TPluginManager *mgr = gROOT->GetPluginManager();
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

  // Write current Storage statistics on a ostream
  void termination(void) const
  {
    std::ostringstream os;
    statsXML(os);
    if (! os.str().empty())
    {
      edm::Service<edm::JobReport> jr;
      jr->reportStorageStats(os.str());
    }
  }

  void stats(std::ostream &o) const
  {
    if (! doStats_)
      return;

    o << "Storage parameters: adaptor: true"
      << " Stats:" << (doStats_ ? "true" : "false") << '\n'
      << " Cache hint:" << cacheHint_ << '\n'
      << " Read hint:" << readHint_ << '\n'
      << "Storage statistics: "
      << StorageAccount::summaryText()
      << "; tfile/read=?/?/" << (TFile::GetFileBytesRead() / 1048576.0) << "MB/?ms/?ms/?ms"
      << "; tfile/write=?/?/" << (TFile::GetFileBytesWritten() / 1048576.0) << "MB/?ms/?ms/?ms";
  }

  void statsXML(std::ostream &o) const
  {
    if (! doStats_)
      return;

    o << "<storage-factory-summary>\n"
      << " <storage-factory-params>\n"
      << "  <param name='enabled' value='true' unit='boolean'/>\n"
      << "  <param name='cache-hint' value='" << cacheHint_ << "' unit='string'/>\n"
      << "  <param name='read-hint' value='" << readHint_ << "' unit='string'/>\n"
      << "  <param name='stats' value='" << (doStats_ ? "true" : "false") << "' unit='boolean'/>\n"
      << " </storage-factory-params>\n"

      << " <storage-factory-stats>\n"
      << StorageAccount::summaryXML() << std::endl
      << "  <storage-root-summary>\n"
      << "   <counter-value subsystem='tfile' counter-name='read' total-megabytes='"
      << (TFile::GetFileBytesRead() / 1048576.0) << "'/>\n"
      << "   <counter-value subsystem='tfile' counter-name='write' total-megabytes='"
      << (TFile::GetFileBytesWritten() / 1048576.0) << "'/>\n"
      << "  </storage-root-summary>\n"
      << " </storage-factory-stats>\n"
      << "</storage-factory-summary>";
  }
};

typedef TFileAdaptor AdaptorConfig;
DEFINE_FWK_SERVICE(AdaptorConfig);
