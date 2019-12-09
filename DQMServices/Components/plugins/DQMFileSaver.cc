#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/JobReport.h"

#include "DataFormats/Histograms/interface/DQMToken.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <TString.h>
#include <TSystem.h>

#include "TFile.h"

#include <boost/format.hpp>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include <string>

namespace saverDetails {
  struct NoCache {};
}  // namespace saverDetails

// NOTE: This module is only save to use in a very restricted set of circumstances:
// - In offline HARVESTING jobs, running single-threaded. RUN and JOB histograms are saved at end of JOB.
// - In multi-run harvesting. JOB histograms are save at end of job.
// - This includes ALCAHARVEST. TODO: check if the data written there is needed for the PCL.
// This module is not used in online. This module is (hopefully?) not used at HLT.
// Online and HLT use modules in DQMServices/FileIO.
class DQMFileSaver : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;
  DQMFileSaver(const edm::ParameterSet &ps);

protected:
  void beginRun(edm::Run const &, edm::EventSetup const &) override{};
  void analyze(const edm::Event &e, const edm::EventSetup &) override;
  void endRun(const edm::Run &, const edm::EventSetup &) override;
  void endJob() override;

private:
  void saveForOffline(const std::string &workflow, int run, int lumi);
  void saveJobReport(const std::string &filename);

  void save(std::string const &filename, uint32_t const run = 0);
  bool createDirectoryIfNeededAndCd(const std::string &path);

  std::string workflow_;
  std::string producer_;
  std::string stream_label_;
  std::string dirName_;
  std::string child_;
  int version_;
  bool runIsComplete_;

  int saveByRun_;
  bool saveAtJobEnd_;
  int forceRunNumber_;

  std::string fileBaseName_;

  DQMStore *dbe_;
  int nrun_;

  // needed only for the harvesting step when saving in the endJob
  int irun_;
};

//--------------------------------------------------------
static void getAnInt(const edm::ParameterSet &ps, int &value, const std::string &name) {
  value = ps.getUntrackedParameter<int>(name, value);
  if (value < 1 && value != -1)
    throw cms::Exception("DQMFileSaver") << "Invalid '" << name << "' parameter '" << value
                                         << "'.  Must be -1 or >= 1.";
}

static std::string onlineOfflineFileName(const std::string &fileBaseName,
                                         const std::string &suffix,
                                         const std::string &workflow,
                                         const std::string &child) {
  size_t pos = 0;
  std::string wflow;
  wflow.reserve(workflow.size() + 3);
  wflow = workflow;
  while ((pos = wflow.find('/', pos)) != std::string::npos)
    wflow.replace(pos++, 1, "__");

  std::string filename = fileBaseName + suffix + wflow + child + ".root";
  return filename;
}

void DQMFileSaver::saveForOffline(const std::string &workflow, int run, int lumi) {
  char suffix[64];
  sprintf(suffix, "R%09d", run);

  std::string filename = onlineOfflineFileName(fileBaseName_, std::string(suffix), workflow, child_);
  assert(lumi == 0);

  // set run end flag
  dbe_->cd();
  dbe_->setCurrentFolder("Info/ProvInfo");

  // do this, because ProvInfo is not yet run in offline DQM
  MonitorElement *me = dbe_->get("Info/ProvInfo/CMSSW");
  if (!me)
    me = dbe_->bookString("CMSSW", edm::getReleaseVersion().c_str());

  me = dbe_->get("Info/ProvInfo/runIsComplete");
  if (!me)
    me = dbe_->bookFloat("runIsComplete");

  if (me) {
    if (runIsComplete_)
      me->Fill(1.);
    else
      me->Fill(0.);
  }

  this->save(filename, run);

  // save the JobReport
  saveJobReport(filename);
}

void DQMFileSaver::saveJobReport(const std::string &filename) {
  // Report the file to job report service.
  edm::Service<edm::JobReport> jr;
  if (jr.isAvailable()) {
    std::map<std::string, std::string> info;
    info["Source"] = "DQMStore";
    info["FileClass"] = "DQM";
    jr->reportAnalysisFile(filename, info);
  }
}

//--------------------------------------------------------
DQMFileSaver::DQMFileSaver(const edm::ParameterSet &ps)
    :

      workflow_(""),
      producer_("DQM"),
      dirName_("."),
      child_(""),
      version_(1),
      runIsComplete_(false),
      saveByRun_(-1),
      saveAtJobEnd_(false),
      forceRunNumber_(-1),
      fileBaseName_(""),
      dbe_(&*edm::Service<DQMStore>()),
      nrun_(0),
      irun_(0) {
  // Note: this is insufficient, we also need to enforce running *after* all
  // DQMEDAnalyzers (a.k.a. EDProducers) in endJob.
  // This is not supported in edm currently.
  consumesMany<DQMToken, edm::InRun>();
  workflow_ = ps.getUntrackedParameter<std::string>("workflow", workflow_);
  if (workflow_.empty() || workflow_[0] != '/' || *workflow_.rbegin() == '/' ||
      std::count(workflow_.begin(), workflow_.end(), '/') != 3 ||
      workflow_.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                  "abcdefghijklmnopqrstuvwxyz"
                                  "0123456789"
                                  "-_/") != std::string::npos)
    throw cms::Exception("DQMFileSaver") << "Invalid 'workflow' parameter '" << workflow_ << "'.  Expected '/A/B/C'.";

  // version number to be used in filename
  // Note that version *must* always be 1 vor DQMGUI upload.
  version_ = ps.getUntrackedParameter<int>("version", version_);
  // flag to signal that file contains data from complete run
  runIsComplete_ = ps.getUntrackedParameter<bool>("runIsComplete", runIsComplete_);

  // Get and check the output directory.
  struct stat s;
  dirName_ = ps.getUntrackedParameter<std::string>("dirName", dirName_);
  if (dirName_.empty() || stat(dirName_.c_str(), &s) == -1)
    throw cms::Exception("DQMFileSaver") << "Invalid 'dirName' parameter '" << dirName_ << "'.";

  // Find out when and how to save files.  The following contraints apply:
  // - For offline allow files to be saved per run, at job end, and run number to be overridden (for mc data).

  getAnInt(ps, saveByRun_, "saveByRun");
  getAnInt(ps, forceRunNumber_, "forceRunNumber");
  saveAtJobEnd_ = ps.getUntrackedParameter<bool>("saveAtJobEnd", saveAtJobEnd_);

  // Set up base file name:
  // - for online and offline, follow the convention <dirName>/<producer>_V<4digits>_
  char version[8];
  sprintf(version, "_V%04d_", int(version_));
  version[7] = '\0';
  fileBaseName_ = dirName_ + "/" + producer_ + version;

  // Log some information what we will do.
  edm::LogInfo("DQMFileSaver") << "DQM file saving settings:\n"
                               << " using base file name '" << fileBaseName_ << "'\n"
                               << " forcing run number " << forceRunNumber_ << "\n"
                               << " saving every " << saveByRun_ << " run(s)\n"
                               << " saving at job end: " << (saveAtJobEnd_ ? "yes" : "no") << "\n";
}

void DQMFileSaver::analyze(const edm::Event &e, const edm::EventSetup &) {
  //save by event and save by time are not supported
  //anymore in the threaded framework. please use
  //savebyLumiSection instead.
}

void DQMFileSaver::endRun(const edm::Run &iRun, const edm::EventSetup &) {
  int irun = iRun.id().run();
  irun_ = irun;
  if (irun > 0 && saveByRun_ > 0 && (nrun_ % saveByRun_) == 0) {
    saveForOffline(workflow_, irun, 0);
  }
}

void DQMFileSaver::endJob() {
  if (saveAtJobEnd_) {
    if (forceRunNumber_ > 0)
      saveForOffline(workflow_, forceRunNumber_, 0);
    else
      saveForOffline(workflow_, irun_, 0);
  }
}

void DQMFileSaver::save(std::string const &filename, uint32_t const run /* = 0 */) {
  // TFile flushes to disk with fsync() on every TDirectory written to
  // the file.  This makes DQM file saving painfully slow, and
  // ironically makes it _more_ likely the file saving gets
  // interrupted and corrupts the file.  The utility class below
  // simply ignores the flush synchronisation.
  class TFileNoSync : public TFile {
  public:
    TFileNoSync(char const *file, char const *opt) : TFile{file, opt} {}
    Int_t SysSync(Int_t) override { return 0; }
  };

  std::cout << "DQMFileSaver::globalEndRun()" << std::endl;

  char suffix[64];
  sprintf(suffix, "R%09d", run);
  TFileNoSync *file = new TFileNoSync(filename.c_str(), "RECREATE");  // open file

  // Traverse all MEs
  auto mes = dbe_->getAllContents("");  // TODO run/lumi and/or job?
  for (auto me : mes) {
    // Modify dirname to comply with DQM GUI format. Change:
    // A/B/C/plot
    // into:
    // DQMData/Run X/A/Run summary/B/C/plot
    std::string dirName = me->getPathname();
    uint64_t firstSlashPos = dirName.find("/");
    if (firstSlashPos == std::string::npos) {
      firstSlashPos = dirName.length();
    }
    dirName = dirName.substr(0, firstSlashPos) + "/Run summary" + dirName.substr(firstSlashPos, dirName.size());
    dirName = "DQMData/Run " + std::to_string(run) + "/" + dirName;

    std::string objectName = me->getName();

    // Create dir if it doesn't exist and cd into it
    createDirectoryIfNeededAndCd(dirName);

    // INTs are saved as strings in this format: <objectName>i=value</objectName>
    // REALs are saved as strings in this format: <objectName>f=value</objectName>
    // STRINGs are saved as strings in this format: <objectName>s="value"</objectName>
    if (me->kind() == MonitorElement::Kind::INT) {
      int value = me->getIntValue();
      std::string content = "<" + objectName + ">i=" + std::to_string(value) + "</" + objectName + ">";
      TObjString str(content.c_str());
      str.Write();
    } else if (me->kind() == MonitorElement::Kind::REAL) {
      double value = me->getFloatValue();
      std::string content = "<" + objectName + ">f=" + std::to_string(value) + "</" + objectName + ">";
      TObjString str(content.c_str());
      str.Write();
    } else if (me->kind() == MonitorElement::Kind::STRING) {
      const std::string &value = me->getStringValue();
      std::string content = "<" + objectName + ">s=\"" + value + "\"</" + objectName + ">";
      TObjString str(content.c_str());
      str.Write();
    } else {
      // Write a histogram
      TH1 *value = me->getTH1();
      value->Write();
    }

    // Go back to the root directory
    gDirectory->cd("/");
  }

  file->Close();
}

// Use this for saving monitoring objects in ROOT files with dir structure;
// cds into directory (creates it first if it doesn't exist);
// returns a success flag
bool DQMFileSaver::createDirectoryIfNeededAndCd(const std::string &path) {
  assert(!path.empty());

  // Find the first path component.
  size_t start = 0;
  size_t end = path.find('/', start);
  if (end == std::string::npos)
    end = path.size();

  while (true) {
    // Check if this subdirectory component exists.  If yes, make sure
    // it is actually a subdirectory.  Otherwise create or cd into it.
    std::string part(path, start, end - start);
    TObject *o = gDirectory->Get(part.c_str());
    if (o && !dynamic_cast<TDirectory *>(o))
      throw cms::Exception("DQMFileSaver") << "Attempt to create directory '" << path
                                           << "' in a file"
                                              " fails because the part '"
                                           << part
                                           << "' already exists and is not"
                                              " directory";
    else if (!o)
      gDirectory->mkdir(part.c_str());

    if (!gDirectory->cd(part.c_str()))
      throw cms::Exception("DQMFileSaver") << "Attempt to create directory '" << path
                                           << "' in a file"
                                              " fails because could not cd into subdirectory '"
                                           << part << "'";

    // Stop if we reached the end, ignoring any trailing '/'.
    if (end + 1 >= path.size())
      break;

    // Find the next path component.
    start = end + 1;
    end = path.find('/', start);
    if (end == std::string::npos)
      end = path.size();
  }

  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMFileSaver);
