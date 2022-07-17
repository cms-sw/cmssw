#include "IOPool/Output/interface/PoolOutputModule.h"

#include "IOPool/Output/src/RootOutputFile.h"

#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "FWCore/Framework/interface/ProductProvenanceRetriever.h"
#include "DataFormats/Provenance/interface/SubProcessParentageHelper.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include "TTree.h"
#include "TBranchElement.h"
#include "TObjArray.h"
#include "RVersion.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include "boost/algorithm/string.hpp"

namespace edm {
  PoolOutputModule::PoolOutputModule(ParameterSet const& pset)
      : edm::one::OutputModuleBase::OutputModuleBase(pset),
        one::OutputModule<WatchInputFiles>(pset),
        rootServiceChecker_(),
        auxItems_(),
        selectedOutputItemList_(),
        fileName_(pset.getUntrackedParameter<std::string>("fileName")),
        logicalFileName_(pset.getUntrackedParameter<std::string>("logicalFileName")),
        catalog_(pset.getUntrackedParameter<std::string>("catalog")),
        maxFileSize_(pset.getUntrackedParameter<int>("maxSize")),
        compressionLevel_(pset.getUntrackedParameter<int>("compressionLevel")),
        compressionAlgorithm_(pset.getUntrackedParameter<std::string>("compressionAlgorithm")),
        basketSize_(pset.getUntrackedParameter<int>("basketSize")),
        eventAuxBasketSize_(pset.getUntrackedParameter<int>("eventAuxiliaryBasketSize")),
        eventAutoFlushSize_(pset.getUntrackedParameter<int>("eventAutoFlushCompressedSize")),
        splitLevel_(std::min<int>(pset.getUntrackedParameter<int>("splitLevel") + 1, 99)),
        basketOrder_(pset.getUntrackedParameter<std::string>("sortBaskets")),
        treeMaxVirtualSize_(pset.getUntrackedParameter<int>("treeMaxVirtualSize")),
        whyNotFastClonable_(pset.getUntrackedParameter<bool>("fastCloning") ? FileBlock::CanFastClone
                                                                            : FileBlock::DisabledInConfigFile),
        dropMetaData_(DropNone),
        moduleLabel_(pset.getParameter<std::string>("@module_label")),
        initializedFromInput_(false),
        outputFileCount_(0),
        inputFileCount_(0),
        branchParents_(),
        branchChildren_(),
        overrideInputFileSplitLevels_(pset.getUntrackedParameter<bool>("overrideInputFileSplitLevels")),
        compactEventAuxiliary_(pset.getUntrackedParameter<bool>("compactEventAuxiliary")),
        mergeJob_(pset.getUntrackedParameter<bool>("mergeJob")),
        rootOutputFile_(),
        statusFileName_(),
        overrideGUID_(pset.getUntrackedParameter<std::string>("overrideGUID")) {
    if (pset.getUntrackedParameter<bool>("writeStatusFile")) {
      std::ostringstream statusfilename;
      statusfilename << moduleLabel_ << '_' << getpid();
      statusFileName_ = statusfilename.str();
    }

    std::string dropMetaData(pset.getUntrackedParameter<std::string>("dropMetaData"));
    if (dropMetaData.empty())
      dropMetaData_ = DropNone;
    else if (dropMetaData == std::string("NONE"))
      dropMetaData_ = DropNone;
    else if (dropMetaData == std::string("DROPPED"))
      dropMetaData_ = DropDroppedPrior;
    else if (dropMetaData == std::string("PRIOR"))
      dropMetaData_ = DropPrior;
    else if (dropMetaData == std::string("ALL"))
      dropMetaData_ = DropAll;
    else {
      throw edm::Exception(errors::Configuration, "Illegal dropMetaData parameter value: ")
          << dropMetaData << ".\n"
          << "Legal values are 'NONE', 'DROPPED', 'PRIOR', and 'ALL'.\n";
    }

    if (!wantAllEvents()) {
      whyNotFastClonable_ += FileBlock::EventSelectionUsed;
    }

    auto const& specialSplit{pset.getUntrackedParameterSetVector("overrideBranchesSplitLevel")};

    specialSplitLevelForBranches_.reserve(specialSplit.size());
    for (auto const& s : specialSplit) {
      specialSplitLevelForBranches_.emplace_back(s.getUntrackedParameter<std::string>("branch"),
                                                 s.getUntrackedParameter<int>("splitLevel"));
    }

    // We don't use this next parameter, but we read it anyway because it is part
    // of the configuration of this module.  An external parser creates the
    // configuration by reading this source code.
    pset.getUntrackedParameterSet("dataset");
  }

  void PoolOutputModule::beginJob() {
    Service<ConstProductRegistry> reg;
    for (auto const& prod : reg->productList()) {
      BranchDescription const& desc = prod.second;
      if (desc.produced() && desc.branchType() == InEvent && !desc.isAlias()) {
        producedBranches_.emplace_back(desc.branchID());
      }
    }
  }

  std::string const& PoolOutputModule::currentFileName() const { return rootOutputFile_->fileName(); }

  PoolOutputModule::AuxItem::AuxItem() : basketSize_(BranchDescription::invalidBasketSize) {}

  PoolOutputModule::OutputItem::OutputItem(BranchDescription const* bd,
                                           EDGetToken const& token,
                                           int splitLevel,
                                           int basketSize)
      : branchDescription_(bd), token_(token), product_(nullptr), splitLevel_(splitLevel), basketSize_(basketSize) {}

  PoolOutputModule::OutputItem::Sorter::Sorter(TTree* tree) : treeMap_(new std::map<std::string, int>) {
    // Fill a map mapping branch names to an index specifying the order in the tree.
    if (tree != nullptr) {
      TObjArray* branches = tree->GetListOfBranches();
      for (int i = 0; i < branches->GetEntries(); ++i) {
        TBranchElement* br = (TBranchElement*)branches->At(i);
        treeMap_->insert(std::make_pair(std::string(br->GetName()), i));
      }
    }
  }

  bool PoolOutputModule::OutputItem::Sorter::operator()(OutputItem const& lh, OutputItem const& rh) const {
    // Provides a comparison for sorting branches according to the index values in treeMap_.
    // Branches not found are always put at the end (i.e. not found > found).
    if (treeMap_->empty())
      return lh < rh;
    std::string const& lstring = lh.branchDescription_->branchName();
    std::string const& rstring = rh.branchDescription_->branchName();
    std::map<std::string, int>::const_iterator lit = treeMap_->find(lstring);
    std::map<std::string, int>::const_iterator rit = treeMap_->find(rstring);
    bool lfound = (lit != treeMap_->end());
    bool rfound = (rit != treeMap_->end());
    if (lfound && rfound) {
      return lit->second < rit->second;
    } else if (lfound) {
      return true;
    } else if (rfound) {
      return false;
    }
    return lh < rh;
  }

  inline bool PoolOutputModule::SpecialSplitLevelForBranch::match(std::string const& iBranchName) const {
    return std::regex_match(iBranchName, branch_);
  }

  std::regex PoolOutputModule::SpecialSplitLevelForBranch::convert(std::string const& iGlobBranchExpression) const {
    std::string tmp(iGlobBranchExpression);
    boost::replace_all(tmp, "*", ".*");
    boost::replace_all(tmp, "?", ".");
    return std::regex(tmp);
  }

  void PoolOutputModule::fillSelectedItemList(BranchType branchType,
                                              std::string const& processName,
                                              TTree* theInputTree,
                                              OutputItemList& outputItemList) {
    SelectedProducts const& keptVector = keptProducts()[branchType];

    if (branchType != InProcess) {
      AuxItem& auxItem = auxItems_[branchType];

      auto basketSize = (InEvent == branchType) ? eventAuxBasketSize_ : basketSize_;

      // Fill AuxItem
      if (theInputTree != nullptr && !overrideInputFileSplitLevels_) {
        TBranch* auxBranch = theInputTree->GetBranch(BranchTypeToAuxiliaryBranchName(branchType).c_str());
        if (auxBranch) {
          auxItem.basketSize_ = auxBranch->GetBasketSize();
        } else {
          auxItem.basketSize_ = basketSize;
        }
      } else {
        auxItem.basketSize_ = basketSize;
      }
    }

    // Fill outputItemList with an entry for each branch.
    for (auto const& kept : keptVector) {
      int splitLevel = BranchDescription::invalidSplitLevel;
      int basketSize = BranchDescription::invalidBasketSize;

      BranchDescription const& prod = *kept.first;
      if (branchType == InProcess && processName != prod.processName()) {
        continue;
      }
      TBranch* theBranch = ((!prod.produced() && theInputTree != nullptr && !overrideInputFileSplitLevels_)
                                ? theInputTree->GetBranch(prod.branchName().c_str())
                                : nullptr);

      if (theBranch != nullptr) {
        splitLevel = theBranch->GetSplitLevel();
        basketSize = theBranch->GetBasketSize();
      } else {
        splitLevel = (prod.splitLevel() == BranchDescription::invalidSplitLevel ? splitLevel_ : prod.splitLevel());
        for (auto const& b : specialSplitLevelForBranches_) {
          if (b.match(prod.branchName())) {
            splitLevel = b.splitLevel_;
          }
        }
        basketSize = (prod.basketSize() == BranchDescription::invalidBasketSize ? basketSize_ : prod.basketSize());
      }
      outputItemList.emplace_back(&prod, kept.second, splitLevel, basketSize);
    }

    // Sort outputItemList to allow fast copying.
    // The branches in outputItemList must be in the same order as in the input tree, with all new branches at the end.
    sort_all(outputItemList, OutputItem::Sorter(theInputTree));
  }

  void PoolOutputModule::beginInputFile(FileBlock const& fb) {
    if (isFileOpen()) {
      //Faster to read ChildrenBranches directly from input
      // file than to build it every event
      auto const& branchToChildMap = fb.branchChildren().childLookup();
      for (auto const& parentToChildren : branchToChildMap) {
        for (auto const& child : parentToChildren.second) {
          branchChildren_.insertChild(parentToChildren.first, child);
        }
      }
      rootOutputFile_->beginInputFile(fb, remainingEvents());
    }
  }

  void PoolOutputModule::openFile(FileBlock const& fb) {
    if (!isFileOpen()) {
      reallyOpenFile();
      beginInputFile(fb);
    }
  }

  void PoolOutputModule::respondToOpenInputFile(FileBlock const& fb) {
    if (!initializedFromInput_) {
      std::vector<std::string> const& processesWithProcessBlockProducts =
          outputProcessBlockHelper().processesWithProcessBlockProducts();
      unsigned int numberOfProcessesWithProcessBlockProducts = processesWithProcessBlockProducts.size();
      unsigned int numberOfTTrees = numberOfRunLumiEventProductTrees + numberOfProcessesWithProcessBlockProducts;
      selectedOutputItemList_.resize(numberOfTTrees);

      for (unsigned int i = InEvent; i < NumBranchTypes; ++i) {
        BranchType branchType = static_cast<BranchType>(i);
        if (branchType != InProcess) {
          std::string processName;
          TTree* theInputTree =
              (branchType == InEvent ? fb.tree() : (branchType == InLumi ? fb.lumiTree() : fb.runTree()));
          OutputItemList& outputItemList = selectedOutputItemList_[branchType];
          fillSelectedItemList(branchType, processName, theInputTree, outputItemList);
        } else {
          // Handle output items in ProcessBlocks
          for (unsigned int k = InProcess; k < numberOfTTrees; ++k) {
            OutputItemList& outputItemList = selectedOutputItemList_[k];
            std::string const& processName = processesWithProcessBlockProducts[k - InProcess];
            TTree* theInputTree = fb.processBlockTree(processName);
            fillSelectedItemList(branchType, processName, theInputTree, outputItemList);
          }
        }
      }
      initializedFromInput_ = true;
    }
    ++inputFileCount_;
    beginInputFile(fb);
  }

  void PoolOutputModule::respondToCloseInputFile(FileBlock const& fb) {
    if (rootOutputFile_)
      rootOutputFile_->respondToCloseInputFile(fb);
  }

  void PoolOutputModule::setProcessesWithSelectedMergeableRunProducts(std::set<std::string> const& processes) {
    processesWithSelectedMergeableRunProducts_.assign(processes.begin(), processes.end());
  }

  PoolOutputModule::~PoolOutputModule() {}

  void PoolOutputModule::write(EventForOutput const& e) {
    updateBranchParents(e);
    rootOutputFile_->writeOne(e);
    if (!statusFileName_.empty()) {
      std::ofstream statusFile(statusFileName_.c_str());
      statusFile << e.id() << " time: " << std::setprecision(3) << TimeOfDay() << '\n';
      statusFile.close();
    }
  }

  void PoolOutputModule::writeLuminosityBlock(LuminosityBlockForOutput const& lb) {
    rootOutputFile_->writeLuminosityBlock(lb);
  }

  void PoolOutputModule::writeRun(RunForOutput const& r) { rootOutputFile_->writeRun(r); }

  void PoolOutputModule::writeProcessBlock(ProcessBlockForOutput const& pb) { rootOutputFile_->writeProcessBlock(pb); }

  void PoolOutputModule::reallyCloseFile() {
    writeEventAuxiliary();
    fillDependencyGraph();
    branchParents_.clear();
    startEndFile();
    writeFileFormatVersion();
    writeFileIdentifier();
    writeIndexIntoFile();
    writeStoredMergeableRunProductMetadata();
    writeProcessHistoryRegistry();
    writeParameterSetRegistry();
    writeProductDescriptionRegistry();
    writeParentageRegistry();
    writeBranchIDListRegistry();
    writeThinnedAssociationsHelper();
    writeProductDependencies();  //branchChildren used here
    writeProcessBlockHelper();
    branchChildren_.clear();
    finishEndFile();

    doExtrasAfterCloseFile();
  }

  // At some later date, we may move functionality from finishEndFile() to here.
  void PoolOutputModule::startEndFile() {}

  void PoolOutputModule::writeFileFormatVersion() { rootOutputFile_->writeFileFormatVersion(); }
  void PoolOutputModule::writeFileIdentifier() { rootOutputFile_->writeFileIdentifier(); }
  void PoolOutputModule::writeIndexIntoFile() { rootOutputFile_->writeIndexIntoFile(); }
  void PoolOutputModule::writeStoredMergeableRunProductMetadata() {
    rootOutputFile_->writeStoredMergeableRunProductMetadata();
  }
  void PoolOutputModule::writeProcessHistoryRegistry() { rootOutputFile_->writeProcessHistoryRegistry(); }
  void PoolOutputModule::writeParameterSetRegistry() { rootOutputFile_->writeParameterSetRegistry(); }
  void PoolOutputModule::writeProductDescriptionRegistry() { rootOutputFile_->writeProductDescriptionRegistry(); }
  void PoolOutputModule::writeParentageRegistry() { rootOutputFile_->writeParentageRegistry(); }
  void PoolOutputModule::writeBranchIDListRegistry() { rootOutputFile_->writeBranchIDListRegistry(); }
  void PoolOutputModule::writeThinnedAssociationsHelper() { rootOutputFile_->writeThinnedAssociationsHelper(); }
  void PoolOutputModule::writeProductDependencies() { rootOutputFile_->writeProductDependencies(); }
  void PoolOutputModule::writeEventAuxiliary() { rootOutputFile_->writeEventAuxiliary(); }
  void PoolOutputModule::writeProcessBlockHelper() { rootOutputFile_->writeProcessBlockHelper(); }
  void PoolOutputModule::finishEndFile() {
    rootOutputFile_->finishEndFile();
    rootOutputFile_ = nullptr;
  }  // propagate_const<T> has no reset() function
  void PoolOutputModule::doExtrasAfterCloseFile() {}
  bool PoolOutputModule::isFileOpen() const { return rootOutputFile_.get() != nullptr; }
  bool PoolOutputModule::shouldWeCloseFile() const { return rootOutputFile_->shouldWeCloseFile(); }

  std::pair<std::string, std::string> PoolOutputModule::physicalAndLogicalNameForNewFile() {
    if (inputFileCount_ == 0) {
      throw edm::Exception(errors::LogicError) << "Attempt to open output file before input file. "
                                               << "Please report this to the core framework developers.\n";
    }
    std::string suffix(".root");
    std::string::size_type offset = fileName().rfind(suffix);
    bool ext = (offset == fileName().size() - suffix.size());
    if (!ext)
      suffix.clear();
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
    ++outputFileCount_;

    return std::make_pair(ofilename.str(), lfilename.str());
  }

  void PoolOutputModule::reallyOpenFile() {
    auto names = physicalAndLogicalNameForNewFile();
    rootOutputFile_ = std::make_unique<RootOutputFile>(this,
                                                       names.first,
                                                       names.second,
                                                       processesWithSelectedMergeableRunProducts_,
                                                       overrideGUID_);  // propagate_const<T> has no reset() function
    // Override the GUID of the first file only, in order to avoid two
    // output files from one Output Module to have identical GUID.
    overrideGUID_.clear();
  }

  void PoolOutputModule::updateBranchParentsForOneBranch(ProductProvenanceRetriever const* provRetriever,
                                                         BranchID const& branchID) {
    ProductProvenance const* provenance = provRetriever->branchIDToProvenanceForProducedOnly(branchID);
    if (provenance != nullptr) {
      BranchParents::iterator it = branchParents_.find(branchID);
      if (it == branchParents_.end()) {
        it = branchParents_.insert(std::make_pair(branchID, std::set<ParentageID>())).first;
      }
      it->second.insert(provenance->parentageID());
    }
  }

  void PoolOutputModule::updateBranchParents(EventForOutput const& e) {
    ProductProvenanceRetriever const* provRetriever = e.productProvenanceRetrieverPtr();
    for (auto const& bid : producedBranches_) {
      updateBranchParentsForOneBranch(provRetriever, bid);
    }
    SubProcessParentageHelper const* helper = subProcessParentageHelper();
    if (helper) {
      for (auto const& bid : subProcessParentageHelper()->producedProducts()) {
        updateBranchParentsForOneBranch(provRetriever, bid);
      }
    }
  }

  void PoolOutputModule::preActionBeforeRunEventAsync(WaitingTaskHolder iTask,
                                                      ModuleCallingContext const& iModuleCallingContext,
                                                      Principal const& iPrincipal) const {
    if (DropAll != dropMetaData_) {
      auto const* ep = dynamic_cast<EventPrincipal const*>(&iPrincipal);
      if (ep) {
        auto pr = ep->productProvenanceRetrieverPtr();
        if (pr) {
          pr->readProvenanceAsync(iTask, &iModuleCallingContext);
        }
      }
    }
  }

  void PoolOutputModule::fillDependencyGraph() {
    for (auto const& branchParent : branchParents_) {
      BranchID const& child = branchParent.first;
      std::set<ParentageID> const& eIds = branchParent.second;
      for (auto const& eId : eIds) {
        Parentage entryDesc;
        ParentageRegistry::instance()->getMapped(eId, entryDesc);
        std::vector<BranchID> const& parents = entryDesc.parents();
        for (auto const& parent : parents) {
          branchChildren_.insertChild(parent, child);
        }
      }
    }
  }

  void PoolOutputModule::fillDescription(ParameterSetDescription& desc) {
    std::string defaultString;

    desc.setComment("Writes runs, lumis, and events into EDM/ROOT files.");
    desc.addUntracked<std::string>("fileName")->setComment("Name of output file.");
    desc.addUntracked<std::string>("logicalFileName", defaultString)
        ->setComment("Passed to job report. Otherwise unused by module.");
    desc.addUntracked<std::string>("catalog", defaultString)
        ->setComment("Passed to job report. Otherwise unused by module.");
    desc.addUntracked<int>("maxSize", 0x7f000000)
        ->setComment(
            "Maximum output file size, in kB.\n"
            "If over maximum, new output file will be started at next input file transition.");
    desc.addUntracked<int>("compressionLevel", 4)->setComment("ROOT compression level of output file.");
    desc.addUntracked<std::string>("compressionAlgorithm", "ZSTD")
        ->setComment(
            "Algorithm used to compress data in the ROOT output file, allowed values are ZLIB, LZMA, LZ4, and ZSTD");
    desc.addUntracked<int>("basketSize", 16384)->setComment("Default ROOT basket size in output file.");
    desc.addUntracked<int>("eventAuxiliaryBasketSize", 16384)
        ->setComment("Default ROOT basket size in output file for EventAuxiliary branch.");
    desc.addUntracked<int>("eventAutoFlushCompressedSize", 20 * 1024 * 1024)
        ->setComment(
            "Set ROOT auto flush stored data size (in bytes) for event TTree. The value sets how large the compressed "
            "buffer is allowed to get. The uncompressed buffer can be quite a bit larger than this depending on the "
            "average compression ratio. The value of -1 just uses ROOT's default value. The value of 0 turns off this "
            "feature.");
    desc.addUntracked<int>("splitLevel", 99)->setComment("Default ROOT branch split level in output file.");
    desc.addUntracked<std::string>("sortBaskets", std::string("sortbasketsbyoffset"))
        ->setComment(
            "Legal values: 'sortbasketsbyoffset', 'sortbasketsbybranch', 'sortbasketsbyentry'.\n"
            "Used by ROOT when fast copying. Affects performance.");
    desc.addUntracked<int>("treeMaxVirtualSize", -1)
        ->setComment("Size of ROOT TTree TBasket cache.  Affects performance.");
    desc.addUntracked<bool>("fastCloning", true)
        ->setComment(
            "True:  Allow fast copying, if possible.\n"
            "False: Disable fast copying.");
    desc.addUntracked("mergeJob", false)
        ->setComment(
            "If set to true and fast copying is disabled, copy input file compression and basket sizes to the output "
            "file.");
    desc.addUntracked<bool>("compactEventAuxiliary", false)
        ->setComment(
            "False: Write EventAuxiliary as we go like any other event metadata branch.\n"
            "True:  Optimize the file layout by deferring writing the EventAuxiliary branch until the output file is "
            "closed.");
    desc.addUntracked<bool>("overrideInputFileSplitLevels", false)
        ->setComment(
            "False: Use branch split levels and basket sizes from input file, if possible.\n"
            "True:  Always use specified or default split levels and basket sizes.");
    desc.addUntracked<bool>("writeStatusFile", false)
        ->setComment("Write a status file. Intended for use by workflow management.");
    desc.addUntracked<std::string>("dropMetaData", defaultString)
        ->setComment(
            "Determines handling of per product per event metadata.  Options are:\n"
            "'NONE':    Keep all of it.\n"
            "'DROPPED': Keep it for products produced in current process and all kept products. Drop it for dropped "
            "products produced in prior processes.\n"
            "'PRIOR':   Keep it for products produced in current process. Drop it for products produced in prior "
            "processes.\n"
            "'ALL':     Drop all of it.");
    desc.addUntracked<std::string>("overrideGUID", defaultString)
        ->setComment(
            "Allows to override the GUID of the file. Intended to be used only in Tier0 for re-creating files.\n"
            "The GUID needs to be of the proper format. If a new output file is started (see maxSize), the GUID of\n"
            "the first file only is overridden, i.e. the subsequent output files have different, generated GUID.");
    {
      ParameterSetDescription dataSet;
      dataSet.setAllowAnything();
      desc.addUntracked<ParameterSetDescription>("dataset", dataSet)
          ->setComment("PSet is only used by Data Operations and not by this module.");
    }
    {
      ParameterSetDescription specialSplit;
      specialSplit.addUntracked<std::string>("branch")->setComment(
          "Name of branch needing a special split level. The name can contain wildcards '*' and '?'");
      specialSplit.addUntracked<int>("splitLevel")->setComment("The special split level for the branch");
      desc.addVPSetUntracked("overrideBranchesSplitLevel", specialSplit, std::vector<ParameterSet>());
    }
    OutputModule::fillDescription(desc);
  }

  void PoolOutputModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    PoolOutputModule::fillDescription(desc);
    descriptions.add("edmOutput", desc);
  }
}  // namespace edm
