#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "TFile.h"
#include "TROOT.h"

using namespace edm;
using namespace std;

TFileService::TFileService(const ParameterSet & cfg, ActivityRegistry & r) :
  TFileDirectory("", "", TFile::Open(cfg.getParameter<string>("fileName").c_str() , "RECREATE"), ""),
  file_(TFileDirectory::file_),
  fileName_(cfg.getParameter<string>("fileName")),
  fileNameRecorded_(false),
  closeFileFast_(cfg.getUntrackedParameter<bool>("closeFileFast", false)) 
{
  // activities to monitor in order to set the proper directory
  r.watchPreModuleConstruction(this, & TFileService::setDirectoryName);
  r.watchPreModule(this, & TFileService::setDirectoryName);
  r.watchPreModuleBeginJob(this, & TFileService::setDirectoryName);
  r.watchPreModuleEndJob(this, & TFileService::setDirectoryName);
  r.watchPreModuleBeginRun(this, & TFileService::setDirectoryName);
  r.watchPreModuleEndRun(this, & TFileService::setDirectoryName);
  r.watchPreModuleBeginLumi(this, & TFileService::setDirectoryName);
  r.watchPreModuleEndLumi(this, & TFileService::setDirectoryName);
  // delay writing into JobReport after BeginJob
  r.watchPostBeginJob(this,&TFileService::afterBeginJob);
}

TFileService::~TFileService() {
  file_->Write();
  if(closeFileFast_) gROOT->GetListOfFiles()->Remove(file_); 
  file_->Close();
  delete file_;
}

void TFileService::setDirectoryName(const ModuleDescription & desc) {
  dir_ = desc.moduleLabel();
  descr_ = (dir_ + " (" + desc.moduleName() + ") folder").c_str();
}

void TFileService::afterBeginJob() {

  if(!fileName_.empty())  {
    if(!fileNameRecorded_) {
      string fullName;
      fullName.reserve(1024);
      fullName = getcwd(&fullName[0],1024);
      fullName += "/" + fileName_;

      map<string,string> fileData;
      fileData.insert(make_pair("Source","TFileService"));

      Service<JobReport> reportSvc;
      reportSvc->reportAnalysisFile(fullName,fileData);
      fileNameRecorded_ = true;
    }
  }
}
