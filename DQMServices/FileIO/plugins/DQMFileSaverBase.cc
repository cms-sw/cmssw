#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"
#include "DataFormats/Histograms/interface/DQMToken.h"

#include "DQMFileSaverBase.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <filesystem>
#include <TString.h>
#include <TSystem.h>

using namespace dqm;

DQMFileSaverBase::DQMFileSaverBase(const edm::ParameterSet &ps) {
  FileParameters fp;

  fp.path_ = ps.getUntrackedParameter<std::string>("path");
  fp.producer_ = ps.getUntrackedParameter<std::string>("producer");
  fp.run_ = 0;
  fp.tag_ = ps.getUntrackedParameter<std::string>("tag");
  fp.lumi_ = 0;
  fp.version_ = 1;
  fp.child_ = "";

  std::unique_lock<std::mutex> lck(initial_fp_lock_);
  initial_fp_ = fp;

  runNumber_ = ps.getUntrackedParameter<int>("runNumber", 111);

  // This makes sure a file saver runs in a very end
  consumesMany<DQMToken, edm::InLumi>();
  consumesMany<DQMToken, edm::InRun>();
}

DQMFileSaverBase::~DQMFileSaverBase() = default;

std::shared_ptr<NoCache> DQMFileSaverBase::globalBeginRun(const edm::Run &r, const edm::EventSetup &) const {
  this->initRun();

  return nullptr;
}

std::shared_ptr<NoCache> DQMFileSaverBase::globalBeginLuminosityBlock(const edm::LuminosityBlock &l,
                                                                      const edm::EventSetup &) const {
  return nullptr;
}

void DQMFileSaverBase::analyze(edm::StreamID, const edm::Event &e, const edm::EventSetup &) const {
  // not supported
}

void DQMFileSaverBase::globalEndLuminosityBlock(const edm::LuminosityBlock &iLS, const edm::EventSetup &) const {
  int ilumi = iLS.id().luminosityBlock();
  int irun = iLS.id().run();

  std::unique_lock<std::mutex> lck(initial_fp_lock_);
  FileParameters fp = initial_fp_;
  lck.unlock();

  fp.lumi_ = ilumi;
  fp.run_ = runNumber_ == 111 ? irun : runNumber_;

  this->saveLumi(fp);
}

void DQMFileSaverBase::globalEndRun(const edm::Run &iRun, const edm::EventSetup &) const {
  std::unique_lock<std::mutex> lck(initial_fp_lock_);
  FileParameters fp = initial_fp_;
  lck.unlock();

  fp.run_ = runNumber_ == 111 ? iRun.id().run() : runNumber_;

  // empty
  this->saveRun(fp);
}

const std::string DQMFileSaverBase::filename(const FileParameters &fp, bool useLumi) {
  char buf[256];
  if (useLumi) {
    snprintf(buf,
             256,
             "%s_V%04d_%s_R%09ld_L%09ld%s",
             fp.producer_.c_str(),
             fp.version_,
             fp.tag_.c_str(),
             fp.run_,
             fp.lumi_,
             fp.child_.c_str());
  } else {
    snprintf(buf,
             256,
             "%s_V%04d_%s_R%09ld%s",
             fp.producer_.c_str(),
             fp.version_,
             fp.tag_.c_str(),
             fp.run_,
             fp.child_.c_str());
  }
  buf[255] = 0;

  namespace fs = std::filesystem;
  fs::path path(fp.path_);
  fs::path file(buf);

  return (path / file).string();
}

void DQMFileSaverBase::saveJobReport(const std::string &filename) const {
  // Report the file to job report service.
  edm::Service<edm::JobReport> jr;
  if (jr.isAvailable()) {
    std::map<std::string, std::string> info;
    info["Source"] = "DQMStore";
    info["FileClass"] = "DQM";
    jr->reportAnalysisFile(filename, info);
  }
}

void DQMFileSaverBase::logFileAction(const std::string &msg, const std::string &fileName) const {
  edm::LogAbsolute("fileAction") << std::setprecision(0) << edm::TimeOfDay() << "  " << msg << fileName;
  edm::FlushMessageLog();
}

void DQMFileSaverBase::fillDescription(edm::ParameterSetDescription &desc) {
  desc.addUntracked<std::string>("tag", "UNKNOWN")->setComment("File tag, DQM_V000_<TAG>*, usually a subsytem name.");

  desc.addUntracked<std::string>("producer", "DQM")
      ->setComment("Base prefix for files, <BASE>_V000_**, either 'DQM' or 'Playback'.");

  desc.addUntracked<std::string>("referenceHandling", "all")->setComment("saveReference_, passed to the DQMStore");

  desc.addUntracked<int>("referenceRequireStatus", dqm::qstatus::STATUS_OK)
      ->setComment("saveReference_, passed to the DQMStore");

  desc.addUntracked<std::string>("path", "./")->setComment("Output path prefix.");

  desc.addUntracked<int>("runNumber", 111)
      ->setComment("Run number passed in the configuration. Will appear in output file names.");
}
