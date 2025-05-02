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
  RootSecondaryFileSequence::RootSecondaryFileSequence(ParameterSet const& pset,
                                                       PoolSource& input,
                                                       InputFileCatalog const& catalog)
      : RootInputFileSequence(pset, catalog),
        input_(input),
        orderedProcessHistoryIDs_(),
        enablePrefetching_(false),
        enforceGUIDInFileName_(pset.getUntrackedParameter<bool>("enforceGUIDInFileName")) {
    // The SiteLocalConfig controls the TTreeCache size and the prefetching settings.
    Service<SiteLocalConfig> pSLC;
    if (pSLC.isAvailable()) {
      enablePrefetching_ = pSLC->enablePrefetching();
    }

    // Prestage the files
    //NOTE: we do not want to stage in all secondary files since we can be given a list of
    // thousands of files and prestaging all those files can cause a site to fail.
    // So, we stage in the first secondary file only.
    setAtFirstFile();
    storage::StorageFactory::get()->stagein(fileNames()[0]);

    // Open the first file.
    for (setAtFirstFile(); !noMoreFiles(); setAtNextFile()) {
      initFile(input_.skipBadFiles());
      if (rootFile())
        break;
    }
    if (rootFile()) {
      input_.productRegistryUpdate().updateFromInput(rootFile()->productRegistry()->productList());
    }
  }

  RootSecondaryFileSequence::~RootSecondaryFileSequence() {}

  void RootSecondaryFileSequence::endJob() { closeFile(); }

  void RootSecondaryFileSequence::closeFile_() {
    // close the currently open file, if any, and delete the RootFile object.
    if (rootFile()) {
      rootFile()->close();
      rootFile().reset();
    }
  }

  void RootSecondaryFileSequence::initFile_(bool skipBadFiles) {
    initTheFile(skipBadFiles, false, nullptr, "secondaryFiles", InputType::SecondaryFile);
  }

  RootSecondaryFileSequence::RootFileSharedPtr RootSecondaryFileSequence::makeRootFile(
      std::shared_ptr<InputFile> filePtr) {
    size_t currentIndexIntoFile = sequenceNumberOfFile();
    return std::make_shared<RootFile>(
        RootFile::FileOptions{.fileName = fileNames()[0],
                              .logicalFileName = logicalFileName(),
                              .filePtr = filePtr,
                              .bypassVersionCheck = input_.bypassVersionCheck(),
                              .enforceGUIDInFileName = enforceGUIDInFileName_},
        InputType::SecondaryFile,
        RootFile::ProcessingOptions{
            .processingMode = input_.processingMode(),
        },
        RootFile::TTreeOptions{.treeMaxVirtualSize = input_.treeMaxVirtualSize(),
                               .enablePrefetching = enablePrefetching_,
                               .promptReading = not input_.delayReadingEventProducts()},
        RootFile::ProductChoices{.productSelectorRules = input_.productSelectorRules(),
                                 .associationsFromSecondary = &associationsFromSecondary_,
                                 .dropDescendantsOfDroppedProducts = input_.dropDescendants(),
                                 .labelRawDataLikeMC = input_.labelRawDataLikeMC()},
        RootFile::CrossFileInfo{.runHelper = input_.runHelper(),
                                .branchIDListHelper = input_.branchIDListHelper(),
                                .thinnedAssociationsHelper = input_.thinnedAssociationsHelper(),
                                .indexesIntoFiles = indexesIntoFiles(),
                                .currentIndexIntoFile = currentIndexIntoFile},
        input_.nStreams(),
        input_.processHistoryRegistryForUpdate(),
        orderedProcessHistoryIDs_);
  }

  void RootSecondaryFileSequence::initAssociationsFromSecondary(std::set<BranchID> const& associationsFromSecondary) {
    for (auto const& branchID : associationsFromSecondary) {
      associationsFromSecondary_.push_back(branchID);
    }
    rootFile()->initAssociationsFromSecondary(associationsFromSecondary_);
  }
}  // namespace edm
