// $Id: PoolOutputModule.cc,v 1.90 2007/11/03 06:53:02 wmtan Exp $

#include "IOPool/Output/src/PoolOutputModule.h"
#include "boost/array.hpp" 
#include "FWCore/MessageLogger/interface/JobReport.h" 
#include "IOPool/Output/src/RootOutputFile.h" 
#include "IOPool/Common/interface/ClassFiller.h"
#include "IOPool/Common/interface/RefStreamer.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include <map>
#include <vector>
#include <iomanip>

namespace edm {
  PoolOutputModule::PoolOutputModule(ParameterSet const& pset) :
    OutputModule(pset),
    fileName_(pset.getUntrackedParameter<std::string>("fileName")),
    logicalFileName_(pset.getUntrackedParameter<std::string>("logicalFileName", std::string())),
    catalog_(pset.getUntrackedParameter<std::string>("catalog", std::string())),
    maxFileSize_(pset.getUntrackedParameter<int>("maxSize", 0x7f000000)),
    compressionLevel_(pset.getUntrackedParameter<int>("compressionLevel", 1)),
    basketSize_(pset.getUntrackedParameter<int>("basketSize", 16384)),
    splitLevel_(pset.getUntrackedParameter<int>("splitLevel", 99)),
    fastCloning_(pset.getUntrackedParameter<bool>("fastCloning", true) && wantAllEvents()),
    fileBlock_(0),
    moduleLabel_(pset.getParameter<std::string>("@module_label")),
    fileCount_(0),
    rootFile_() {
    ClassFiller();
    // We need to set a custom streamer for edm::RefCore so that it will not be split.
    // even though a custom streamer is not otherwise necessary.
    SetRefStreamer();
    // We don't use this next parameter, but we read it anyway because it is part
    // of the configuration of this module.  An external parser creates the
    // configuration by reading this source code.
    pset.getUntrackedParameter<ParameterSet>("dataset", ParameterSet());
  }

  void PoolOutputModule::beginInputFile(FileBlock const& fb) {
    fileBlock_ = const_cast<FileBlock *>(&fb);
    if (!isFileOpen()) {
      // Since we currently merge files, if the first input file is not fast clonable,
      // we cannot do fast cloning at all.
      if (!fb.fastClonable()) {
        fastCloning_ = false;
      }
      doOpenFile();
    }
    rootFile_->beginInputFile(fb);
  }

  void PoolOutputModule::beginJob(EventSetup const&) {
  }

  void PoolOutputModule::endJob() {
    // doEndFile should be called from the framework, not internally...
//     if (rootFile_.get() != 0) {
//       rootFile_->endFile();
//       rootFile_.reset();
//     }
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

  void PoolOutputModule::endRun(RunPrincipal const& r) {
      if (hasNewlyDroppedBranch()[InRun]) r.addToProcessHistory();
      if (rootFile_->writeRun(r)) {
	// maybeEndFile should be called from the framework, not internally
	// rootFile_->endFile();
	// rootFile_.reset();
      }
  }

  // At some later date, we may move functionality from finishEndFile() to here.
  void PoolOutputModule::startEndFile() { }

  void PoolOutputModule::writeFileFormatVersion() { rootFile_->writeFileFormatVersion(); }
  void PoolOutputModule::writeFileIdentifier() { rootFile_->writeFileIdentifier(); }
  void PoolOutputModule::writeProcessConfigurationRegistry() { rootFile_->writeProcessConfigurationRegistry(); }
  void PoolOutputModule::writeProcessHistoryRegistry() { rootFile_->writeProcessHistoryRegistry(); }
  void PoolOutputModule::writeModuleDescriptionRegistry() { rootFile_->writeModuleDescriptionRegistry(); }
  void PoolOutputModule::writeParameterSetRegistry() { rootFile_->writeParameterSetRegistry(); }
  void PoolOutputModule::writeProductDescriptionRegistry() { rootFile_->writeProductDescriptionRegistry(); }
  void PoolOutputModule::finishEndFile() { rootFile_->finishEndFile(); rootFile_.reset(); }
  bool PoolOutputModule::isFileOpen() const { return rootFile_.get() != 0; }


  bool PoolOutputModule::isFileFull() const { return rootFile_->isFileFull(); }

  void PoolOutputModule::doOpenFile() {
      if (fileBlock_ == 0) {
        throw edm::Exception(edm::errors::LogicError)
          << "Attempt to open output file before input file. "
          << "Please report this to the core framework developers.\n";
      }
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
