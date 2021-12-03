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
    StorageFactory::get()->stagein(fileName());

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

  void RootSecondaryFileSequence::endJob() { closeFile_(); }

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
    return std::make_shared<RootFile>(fileName(),
                                      input_.processConfiguration(),
                                      logicalFileName(),
                                      filePtr,
                                      input_.nStreams(),
                                      input_.treeMaxVirtualSize(),
                                      input_.processingMode(),
                                      input_.runHelper(),
                                      input_.productSelectorRules(),
                                      InputType::SecondaryFile,
                                      input_.branchIDListHelper(),
                                      input_.thinnedAssociationsHelper(),
                                      &associationsFromSecondary_,
                                      input_.dropDescendants(),
                                      input_.processHistoryRegistryForUpdate(),
                                      indexesIntoFiles(),
                                      currentIndexIntoFile,
                                      orderedProcessHistoryIDs_,
                                      input_.bypassVersionCheck(),
                                      input_.labelRawDataLikeMC(),
                                      enablePrefetching_,
                                      enforceGUIDInFileName_);
  }

  void RootSecondaryFileSequence::initAssociationsFromSecondary(std::set<BranchID> const& associationsFromSecondary) {
    for (auto const& branchID : associationsFromSecondary) {
      associationsFromSecondary_.push_back(branchID);
    }
    rootFile()->initAssociationsFromSecondary(associationsFromSecondary_);
  }
}  // namespace edm
