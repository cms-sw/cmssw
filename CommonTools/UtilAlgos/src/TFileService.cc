#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "TFile.h"
#include "TROOT.h"

#include <map>
#include <unistd.h>

const std::string TFileService::kSharedResource = "TFileService";
thread_local TFileDirectory TFileService::tFileDirectory_;

TFileService::TFileService(const edm::ParameterSet & cfg, edm::ActivityRegistry & r) :
  file_(nullptr),
  fileName_(cfg.getParameter<std::string>("fileName")),
  fileNameRecorded_(false),
  closeFileFast_(cfg.getUntrackedParameter<bool>("closeFileFast", false)) 
{
  tFileDirectory_ = TFileDirectory("",
                                  "",
                                  TFile::Open(fileName_.c_str(), "RECREATE"),
                                  "");
  file_ = tFileDirectory_.file_;

  // activities to monitor in order to set the proper directory
  r.watchPreModuleConstruction(this, & TFileService::setDirectoryName);
  r.watchPreModuleBeginJob(this, & TFileService::setDirectoryName);
  r.watchPreModuleEndJob(this, & TFileService::setDirectoryName);
  r.watchPreModuleEvent(this, & TFileService::preModuleEvent);
  r.watchPostModuleEvent(this, & TFileService::postModuleEvent);

  r.watchPreModuleGlobalBeginRun(this, &TFileService::preModuleGlobal);
  r.watchPostModuleGlobalBeginRun(this, &TFileService::postModuleGlobal);
  r.watchPreModuleGlobalEndRun(this, &TFileService::preModuleGlobal);
  r.watchPostModuleGlobalEndRun(this, &TFileService::postModuleGlobal);

  r.watchPreModuleGlobalBeginLumi(this, &TFileService::preModuleGlobal);
  r.watchPostModuleGlobalBeginLumi(this, &TFileService::postModuleGlobal);
  r.watchPreModuleGlobalEndLumi(this, &TFileService::preModuleGlobal);
  r.watchPostModuleGlobalEndLumi(this, &TFileService::postModuleGlobal);

  // delay writing into JobReport after BeginJob
  r.watchPostBeginJob(this, &TFileService::afterBeginJob);
}

TFileService::~TFileService() {
  file_->Write();
  if(closeFileFast_) gROOT->GetListOfFiles()->Remove(file_); 
  file_->Close();
  delete file_;
}

void TFileService::setDirectoryName(const edm::ModuleDescription & desc) {
  tFileDirectory_.file_ = file_;
  tFileDirectory_.dir_ = desc.moduleLabel();
  tFileDirectory_.descr_ = (tFileDirectory_.dir_ + " (" + desc.moduleName() + ") folder").c_str();
}

void TFileService::preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
  setDirectoryName(*mcc.moduleDescription());
}

void TFileService::postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
  edm::ModuleCallingContext const* previous_mcc = mcc.previousModuleOnThread();
  if(previous_mcc) {
    setDirectoryName(*previous_mcc->moduleDescription());
  }
}

void
TFileService::preModuleGlobal(edm::GlobalContext const&, edm::ModuleCallingContext const& mcc) {
  setDirectoryName(*mcc.moduleDescription());
}

void
TFileService::postModuleGlobal(edm::GlobalContext const&, edm::ModuleCallingContext const& mcc) {
  edm::ModuleCallingContext const* previous_mcc = mcc.previousModuleOnThread();
  if(previous_mcc) {
    setDirectoryName(*previous_mcc->moduleDescription());
  }
}

void TFileService::afterBeginJob() {

  if(!fileName_.empty())  {
    if(!fileNameRecorded_) {
      std::string fullName;
      fullName.reserve(1024);
      fullName = getcwd(&fullName[0],1024);
      fullName += "/" + fileName_;

      std::map<std::string, std::string> fileData;
      fileData.insert(std::make_pair("Source","TFileService"));

      edm::Service<edm::JobReport> reportSvc;
      reportSvc->reportAnalysisFile(fullName,fileData);
      fileNameRecorded_ = true;
    }
  }
}
