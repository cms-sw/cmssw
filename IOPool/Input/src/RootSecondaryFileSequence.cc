/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "DuplicateChecker.h"
#include "PoolSource.h"
#include "InputFile.h"
#include "RootFile.h"
#include "RootSecondaryFileSequence.h"
#include "RootTree.h"

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"

namespace edm {
  RootSecondaryFileSequence::RootSecondaryFileSequence(
                ParameterSet const& pset,
                PoolSource& input,
                InputFileCatalog const& catalog,
                unsigned int nStreams) :
    RootInputFileSequence(pset, catalog),
    input_(input),
    firstFile_(true),
    orderedProcessHistoryIDs_(),
    nStreams_(nStreams),
    // The default value provided as the second argument to the getUntrackedParameter function call
    // is not used when the ParameterSet has been validated and the parameters are not optional
    // in the description.  This is currently true when PoolSource is the primary input source.
    // The modules that use PoolSource as a SecSource have not defined their fillDescriptions function
    // yet, so the ParameterSet does not get validated yet.  As soon as all the modules with a SecSource
    // have defined descriptions, the defaults in the getUntrackedParameterSet function calls can
    // and should be deleted from the code.
    skipBadFiles_(pset.getUntrackedParameter<bool>("skipBadFiles", false)),
    bypassVersionCheck_(pset.getUntrackedParameter<bool>("bypassVersionCheck", false)),
    treeMaxVirtualSize_(pset.getUntrackedParameter<int>("treeMaxVirtualSize", -1)),
    setRun_(pset.getUntrackedParameter<unsigned int>("setRunNumber", 0U)),
    productSelectorRules_(pset, "inputCommands", "InputSource"),
    dropDescendants_(pset.getUntrackedParameter<bool>("dropDescendantsOfDroppedBranches", true)),
    labelRawDataLikeMC_(pset.getUntrackedParameter<bool>("labelRawDataLikeMC", true)),
    enablePrefetching_(false) {

    // The SiteLocalConfig controls the TTreeCache size and the prefetching settings.
    Service<SiteLocalConfig> pSLC;
    if(pSLC.isAvailable()) {
      enablePrefetching_ = pSLC->enablePrefetching();
    }

    // Prestage the files
    //NOTE: we do not want to stage in all secondary files since we can be given a list of
    // thousands of files and prestaging all those files can cause a site to fail.
    // So, we stage in the first secondary file only.
    setAtFirstFile();
    StorageFactory::get()->stagein(fileName());

    // Open the first file.
    for(setAtFirstFile(); !noMoreFiles(); setAtNextFile()) {
      initFile(skipBadFiles_);
      if(rootFile()) break;
    }
    if(rootFile()) {
      input_.productRegistryUpdate().updateFromInput(rootFile()->productRegistry()->productList());
    }
  }

  RootSecondaryFileSequence::~RootSecondaryFileSequence() {
  }

  void
  RootSecondaryFileSequence::endJob() {
    closeFile_();
  }

  void
  RootSecondaryFileSequence::closeFile_() {
    // close the currently open file, if any, and delete the RootFile object.
    if(rootFile()) {
      rootFile()->close();
      rootFile().reset();
    }
  }

  void RootSecondaryFileSequence::initFile_(bool skipBadFiles) {
    initTheFile(skipBadFiles, false, nullptr, "secondaryFiles", InputType::SecondaryFile);
  }

  RootSecondaryFileSequence::RootFileSharedPtr
  RootSecondaryFileSequence::makeRootFile(std::shared_ptr<InputFile> filePtr) {
    size_t currentIndexIntoFile = sequenceNumberOfFile();
    return std::make_shared<RootFile>(
          fileName(),
          input_.processConfiguration(),
          logicalFileName(),
          filePtr,
          nullptr, // eventSkipperByID_
          false,   // initialNumberOfEventsToSkip_ != 0
          -1,      // remainingEvents() 
          -1,      // remainingLuminosityBlocks()
	  nStreams_,
          0U,      // treeCacheSize_
          treeMaxVirtualSize_,
          input_.processingMode(),
          setRun_,
          false,   // noEventSort_
          productSelectorRules_,
          InputType::SecondaryFile,
          input_.branchIDListHelper(),
          input_.thinnedAssociationsHelper(),
          associationsFromSecondary_,
          nullptr, //duplicateChecker_
          dropDescendants_,
          input_.processHistoryRegistryForUpdate(),
          indexesIntoFiles(),
          currentIndexIntoFile,
          orderedProcessHistoryIDs_,
          bypassVersionCheck_,
          labelRawDataLikeMC_,
          false,   // usingGoToEvent_
          enablePrefetching_);
  }

  void
  RootSecondaryFileSequence::initAssociationsFromSecondary(std::set<BranchID> const& associationsFromSecondary) {
    for(auto const& branchID : associationsFromSecondary) {
      associationsFromSecondary_.push_back(branchID);
    }
    rootFile()->initAssociationsFromSecondary(associationsFromSecondary_);
  }
}
