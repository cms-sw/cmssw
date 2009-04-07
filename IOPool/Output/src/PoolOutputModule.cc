#include "IOPool/Output/src/PoolOutputModule.h"

#include "FWCore/MessageLogger/interface/JobReport.h" 
#include "IOPool/Output/src/RootOutputFile.h" 

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "TTree.h"
#include "TBranchElement.h"
#include "TObjArray.h"

#include <iomanip>
#include <sstream>

namespace edm {
  PoolOutputModule::PoolOutputModule(ParameterSet const& pset) :
    OutputModule(pset),
    rootServiceChecker_(),
    selectedOutputItemList_(), 
    fileName_(pset.getUntrackedParameter<std::string>("fileName")),
    logicalFileName_(pset.getUntrackedParameter<std::string>("logicalFileName", std::string())),
    catalog_(pset.getUntrackedParameter<std::string>("catalog", std::string())),
    maxFileSize_(pset.getUntrackedParameter<int>("maxSize", 0x7f000000)),
    compressionLevel_(pset.getUntrackedParameter<int>("compressionLevel", 7)),
    basketSize_(pset.getUntrackedParameter<int>("basketSize", 16384)),
    splitLevel_(pset.getUntrackedParameter<int>("splitLevel", 99)),
    treeMaxVirtualSize_(pset.getUntrackedParameter<int>("treeMaxVirtualSize", -1)),
    fastCloning_(pset.getUntrackedParameter<bool>("fastCloning", true) && wantAllEvents()),
    dropMetaData_(DropNone),
    moduleLabel_(pset.getParameter<std::string>("@module_label")),
    outputFileCount_(0),
    inputFileCount_(0),
    rootOutputFile_() {
      std::string dropMetaData(pset.getUntrackedParameter<std::string>("dropMetaData", std::string()));
      if (dropMetaData.empty()) dropMetaData_ = DropNone;
      else if (dropMetaData == std::string("NONE")) dropMetaData_ = DropNone;
      else if (dropMetaData == std::string("DROPPED")) dropMetaData_ = DropDroppedPrior;
      else if (dropMetaData == std::string("PRIOR")) dropMetaData_ = DropPrior;
      else if (dropMetaData == std::string("ALL")) dropMetaData_ = DropAll;
      else {
        throw edm::Exception(errors::Configuration, "Illegal dropMetaData parameter value: ")
            << dropMetaData << ".\n"
            << "Legal values are 'NONE', 'DROPPED', 'PRIOR', and 'ALL'.\n";
      }

    // We don't use this next parameter, but we read it anyway because it is part
    // of the configuration of this module.  An external parser creates the
    // configuration by reading this source code.
    pset.getUntrackedParameter<ParameterSet>("dataset", ParameterSet());
  }

  PoolOutputModule::OutputItem::Sorter::Sorter(TTree * tree) {
    // Fill a map mapping branch names to an index specifying the order in the tree.
    if (tree != 0) {
      TObjArray * branches = tree->GetListOfBranches();
      for (int i = 0; i < branches->GetEntries(); ++i) {
        TBranchElement * br = (TBranchElement *)branches->At(i);
        treeMap_.insert(std::make_pair(std::string(br->GetName()), i));
      }
    }
  }

  bool
  PoolOutputModule::OutputItem::Sorter::operator()(OutputItem const& lh, OutputItem const& rh) const {
    // Provides a comparison for sorting branches according to the index values in treeMap_.
    // Branches not found are always put at the end (i.e. not found > found).
    if (treeMap_.empty()) return lh < rh;
    std::string const& lstring = lh.branchDescription_->branchName();
    std::string const& rstring = rh.branchDescription_->branchName();
    std::map<std::string, int>::const_iterator lit = treeMap_.find(lstring);
    std::map<std::string, int>::const_iterator rit = treeMap_.find(rstring);
    bool lfound = (lit != treeMap_.end());
    bool rfound = (rit != treeMap_.end());
    if (lfound && rfound) {
      return lit->second < rit->second;
    } else if (lfound) {
      return true;
    } else if (rfound) {
      return false;
    }
    return lh < rh;
  }

  void PoolOutputModule::fillSelectedItemList(BranchType branchType, TTree * theTree) {

    Selections const& keptVector =    keptProducts()[branchType];
    OutputItemList&   outputItemList = selectedOutputItemList_[branchType];

    // Fill outputItemList with an entry for each branch.
    for (Selections::const_iterator it = keptVector.begin(), itEnd = keptVector.end(); it != itEnd; ++it) {
      BranchDescription const& prod = **it;
      outputItemList.push_back(OutputItem(&prod));
    }

    // Sort outputItemList to allow fast copying.
    // The branches in outputItemList must be in the same order as in the input tree, with all new branches at the end.
    sort_all(outputItemList, OutputItem::Sorter(theTree));
  }

  void PoolOutputModule::openFile(FileBlock const& fb) {
    if (!isFileOpen()) {
      if (fb.tree() == 0) {
	fastCloning_ = false;
      }
      doOpenFile();
      respondToOpenInputFile(fb);
    }
  }

  void PoolOutputModule::respondToOpenInputFile(FileBlock const& fb) {
    for (int i = InEvent; i < NumBranchTypes; ++i) {
      BranchType branchType = static_cast<BranchType>(i);
      if (inputFileCount_ == 0) {
        TTree * theTree = (branchType == InEvent ? fb.tree() : 
		          (branchType == InLumi ? fb.lumiTree() :
                          fb.runTree()));
        fillSelectedItemList(branchType, theTree);
      }
    }
    ++inputFileCount_;
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
      Service<JobReport> reportSvc;
      reportSvc->reportRunNumber(r.run());
  }

  // At some later date, we may move functionality from finishEndFile() to here.
  void PoolOutputModule::startEndFile() { }


  void PoolOutputModule::writeFileFormatVersion() { rootOutputFile_->writeFileFormatVersion(); }
  void PoolOutputModule::writeFileIdentifier() { rootOutputFile_->writeFileIdentifier(); }
  void PoolOutputModule::writeFileIndex() { rootOutputFile_->writeFileIndex(); }
  void PoolOutputModule::writeEventHistory() { rootOutputFile_->writeEventHistory(); }
  void PoolOutputModule::writeProcessConfigurationRegistry() { rootOutputFile_->writeProcessConfigurationRegistry(); }
  void PoolOutputModule::writeProcessHistoryRegistry() { rootOutputFile_->writeProcessHistoryRegistry(); }
  void PoolOutputModule::writeParameterSetRegistry() { rootOutputFile_->writeParameterSetRegistry(); }
  void PoolOutputModule::writeProductDescriptionRegistry() { rootOutputFile_->writeProductDescriptionRegistry(); }
  void PoolOutputModule::writeParentageRegistry() { rootOutputFile_->writeParentageRegistry(); }
  void PoolOutputModule::writeBranchIDListRegistry() { rootOutputFile_->writeBranchIDListRegistry(); }
  void PoolOutputModule::writeProductDependencies() { rootOutputFile_->writeProductDependencies(); }
  void PoolOutputModule::finishEndFile() { rootOutputFile_->finishEndFile(); rootOutputFile_.reset(); }
  bool PoolOutputModule::isFileOpen() const { return rootOutputFile_.get() != 0; }
  bool PoolOutputModule::shouldWeCloseFile() const { return rootOutputFile_->shouldWeCloseFile(); }

  void PoolOutputModule::doOpenFile() {
      if (inputFileCount_ == 0) {
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
      if (outputFileCount_) {
        ofilename << std::setw(3) << std::setfill('0') << outputFileCount_;
	if (!logicalFileName().empty()) {
	  lfilename << std::setw(3) << std::setfill('0') << outputFileCount_;
	}
      }
      ofilename << suffix;
      rootOutputFile_.reset(new RootOutputFile(this, ofilename.str(), lfilename.str()));
      ++outputFileCount_;
  }
}
