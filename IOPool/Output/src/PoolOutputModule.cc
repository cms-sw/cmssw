// $Id: PoolOutputModule.cc,v 1.81 2007/08/21 00:03:13 wmtan Exp $

#include "IOPool/Output/src/PoolOutputModule.h"
#include "boost/array.hpp" 
#include "FWCore/MessageLogger/interface/JobReport.h" 
#include "IOPool/Output/src/RootOutputFile.h" 
#include "IOPool/Common/interface/ClassFiller.h"
#include "IOPool/Common/interface/RefStreamer.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include <map>
#include <vector>
#include <iomanip>

namespace edm {
  PoolOutputModule::PoolOutputModule(ParameterSet const& pset) :
    OutputModule(pset),
    catalog_(pset),
    autoSaveInterval_(pset.getUntrackedParameter<unsigned int>("autoSaveInterval", 100U)),
    maxFileSize_(pset.getUntrackedParameter<int>("maxSize", 0x7f000000)),
    compressionLevel_(pset.getUntrackedParameter<int>("compressionLevel", 1)),
    basketSize_(pset.getUntrackedParameter<int>("basketSize", 16384)),
    splitLevel_(pset.getUntrackedParameter<int>("splitLevel", 99)),
    moduleLabel_(pset.getParameter<std::string>("@module_label")),
    fileCount_(0),
    rootFile_() {
    ClassFiller();
    // We need to set a custom streamer for edm::RefCore so that it will not be split.
    // even though a custom streamer is not otherwise necessary.
    SetRefStreamer();
  }

  void PoolOutputModule::beginJob(EventSetup const&) {
  }

  void PoolOutputModule::endJob() {
    if (rootFile_.get() != 0) {
      rootFile_->endFile();
      rootFile_.reset();
    }
  }

  PoolOutputModule::~PoolOutputModule() {
  }

  void PoolOutputModule::write(EventPrincipal const& e) {
      if (hasNewlyDroppedBranch()[InEvent]) e.addToProcessHistory();
      rootFile_->writeOne(e);
  }

  void PoolOutputModule::endLuminosityBlock(LuminosityBlockPrincipal const& lb) {
      if (hasNewlyDroppedBranch()[InLumi]) lb.addToProcessHistory();
      rootFile_->writeLuminosityBlock(lb);
      Service<JobReport> reportSvc;
      reportSvc->reportLumiSection(lb.id().run(), lb.id().luminosityBlock());
  }

  void PoolOutputModule::beginRun(RunPrincipal const&) {
    if (rootFile_.get() == 0) {
      std::string suffix(".root");
      std::string::size_type offset = fileName().rfind(suffix);
      bool ext = (offset == fileName().size() - suffix.size());
      if (!ext) suffix.clear();
      std::string fileBase(ext ? fileName().substr(0, offset) : fileName());
      std::ostringstream ofilename;
      std::ostringstream lfilename;
      ofilename << fileBase;
      lfilename << logicalFileName();
      if (fileCount_) {
        ofilename << std::setw(3) << std::setfill('0') << fileCount_;
	if (!logicalFileName().empty()) {
	  lfilename << std::setw(3) << std::setfill('0') << fileCount_;
	}
      }
      ofilename << suffix;
      rootFile_ = boost::shared_ptr<RootOutputFile>(new RootOutputFile(this, ofilename.str(), lfilename.str()));
      ++fileCount_;
    }
  }

  void PoolOutputModule::endRun(RunPrincipal const& r) {
      if (hasNewlyDroppedBranch()[InRun]) r.addToProcessHistory();
      if (rootFile_->writeRun(r)) {
	rootFile_->endFile();
	rootFile_.reset();
      }
  }
}
