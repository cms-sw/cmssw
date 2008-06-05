#include "IOPool/TFileAdaptor/interface/TStorageFactoryFile.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
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
  std::string mode_;
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
      doStats_(true),
      mode_("none")
  {
    if (! (enabled_ = p.getUntrackedParameter<bool> ("enable", enabled_)))
      return;

    doStats_ = p.getUntrackedParameter<bool> ("stats", doStats_);
    mode_ = p.getUntrackedParameter<std::string> ("mode", mode_);
    native_ = p.getUntrackedParameter<std::vector<std::string> >("native", native_);
    ar.watchPostEndJob(this, &TFileAdaptor::termination);

    // Handle standard modes (none available right now).
    //   default = adaptor:on stats:on
    if (mode_ == "default")
      doStats_ = true;

    // enable file access stats accounting if requested
    StorageFactory::get()->enableAccounting(doStats_);

    // set our own root plugins
    TPluginManager *mgr = gROOT->GetPluginManager();
    mgr->LoadHandlersFromPluginDirs();

    if (!native("file"))    addType(mgr, "^file:");
    if (!native("http"))    addType(mgr, "^http:");
    if (!native("ftp"))     addType(mgr, "^ftp:");
    /* always */            addType(mgr, "^web:");
    /* always */            addType(mgr, "^gsiftp:");
    /* always */            addType(mgr, "^sfn:");
    if (!native("rfio"))    addType(mgr, "^rfio:");
    if (!native("dcache"))  addType(mgr, "^dcache:");
    if (!native("dcap"))    addType(mgr, "^dcap:");
    if (!native("gsidcap")) addType(mgr, "^gsidcap:");
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
