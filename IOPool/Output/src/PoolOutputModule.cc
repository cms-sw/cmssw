#include "IOPool/Output/src/PoolOutputModule.h"

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
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "TTree.h"

#include <iomanip>
#include <sstream>

namespace edm {
  PoolOutputModule::PoolOutputModule(ParameterSet const& pset) :
    OutputModule(pset),
    fileName_(pset.getUntrackedParameter<std::string>("fileName")),
    logicalFileName_(pset.getUntrackedParameter<std::string>("logicalFileName", std::string())),
    catalog_(pset.getUntrackedParameter<std::string>("catalog", std::string())),
    maxFileSize_(pset.getUntrackedParameter<int>("maxSize", 0x7f000000)),
    compressionLevel_(pset.getUntrackedParameter<int>("compressionLevel", 7)),
    basketSize_(pset.getUntrackedParameter<int>("basketSize", 16384)),
    splitLevel_(pset.getUntrackedParameter<int>("splitLevel", 99)),
    treeMaxVirtualSize_(pset.getUntrackedParameter<int>("treeMaxVirtualSize", -1)),
    fastCloning_(pset.getUntrackedParameter<bool>("fastCloning", true) && wantAllEvents()),
    fileBlock_(0),
    moduleLabel_(pset.getParameter<std::string>("@module_label")),
    fileCount_(0),
    rootOutputFile_() {
    ClassFiller();
    // We need to set a custom streamer for edm::RefCore so that it will not be split.
    // even though a custom streamer is not otherwise necessary.
    SetRefStreamer();
    // We don't use this next parameter, but we read it anyway because it is part
    // of the configuration of this module.  An external parser creates the
    // configuration by reading this source code.
    pset.getUntrackedParameter<ParameterSet>("dataset", ParameterSet());
  }

  void PoolOutputModule::openFile(FileBlock const& fb) {
    if (!isFileOpen()) {
      if (fb.tree() == 0 || fb.fileFormatVersion().value_ < 8) {
	fastCloning_ = false;
      }
      doOpenFile();
      respondToOpenInputFile(fb);
    }
  }

  void PoolOutputModule::respondToOpenInputFile(FileBlock const& fb) {
    fileBlock_ = &fb;
    if (isFileOpen()) {
      bool fastCloneThisOne = fb.tree() != 0 &&
                            (remainingEvents() < 0 || remainingEvents() >= fb.tree()->GetEntries());
      rootOutputFile_->beginInputFile(fb, fastCloneThisOne && fastCloning_);
    }
  }

  void PoolOutputModule::respondToCloseInputFile(FileBlock const& fb) {
    if (rootOutputFile_) rootOutputFile_->respondToCloseInputFile(fb);
  }

  PoolOutputModule::~PoolOutputModule() {
  }

  void PoolOutputModule::write(EventPrincipal const& e) {
      if (hasNewlyDroppedBranch()[InEvent]) e.addToProcessHistory();
      rootOutputFile_->writeOne(e);
  }

  void PoolOutputModule::writeLuminosityBlock(LuminosityBlockPrincipal const& lb) {
      if (hasNewlyDroppedBranch()[InLumi]) lb.addToProcessHistory();
      rootOutputFile_->writeLuminosityBlock(lb);
      Service<JobReport> reportSvc;
      reportSvc->reportLumiSection(lb.id().run(), lb.id().luminosityBlock());
  }

  void PoolOutputModule::writeRun(RunPrincipal const& r) {
      if (hasNewlyDroppedBranch()[InRun]) r.addToProcessHistory();
      rootOutputFile_->writeRun(r);
  }

  // At some later date, we may move functionality from finishEndFile() to here.
  void PoolOutputModule::startEndFile() { }


  void PoolOutputModule::writeFileFormatVersion() { rootOutputFile_->writeFileFormatVersion(); }
  void PoolOutputModule::writeFileIdentifier() { rootOutputFile_->writeFileIdentifier(); }
  void PoolOutputModule::writeFileIndex() { rootOutputFile_->writeFileIndex(); }
  void PoolOutputModule::writeEventHistory() { rootOutputFile_->writeEventHistory(); }
  void PoolOutputModule::writeProcessConfigurationRegistry() { rootOutputFile_->writeProcessConfigurationRegistry(); }
  void PoolOutputModule::writeProcessHistoryRegistry() { rootOutputFile_->writeProcessHistoryRegistry(); }
  void PoolOutputModule::writeModuleDescriptionRegistry() { rootOutputFile_->writeModuleDescriptionRegistry(); }
  void PoolOutputModule::writeParameterSetRegistry() { rootOutputFile_->writeParameterSetRegistry(); }
  void PoolOutputModule::writeProductDescriptionRegistry() { rootOutputFile_->writeProductDescriptionRegistry(); }
  void PoolOutputModule::writeProductDependencies() { rootOutputFile_->writeProductDependencies(); }
  void PoolOutputModule::writeEntryDescriptions() { rootOutputFile_->writeEntryDescriptions(); }
  void PoolOutputModule::finishEndFile() { rootOutputFile_->finishEndFile(); rootOutputFile_.reset(); }
  bool PoolOutputModule::isFileOpen() const { return rootOutputFile_.get() != 0; }
  bool PoolOutputModule::shouldWeCloseFile() const { return rootOutputFile_->shouldWeCloseFile(); }

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
      rootOutputFile_.reset(new RootOutputFile(this, ofilename.str(), lfilename.str()));
      ++fileCount_;
  }
}
