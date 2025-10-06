#include "FWIO/RNTupleTempOutput/interface/RNTupleTempOutputModule.h"

#include "FWIO/RNTupleTempOutput/src/RootOutputFile.h"

#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "FWCore/Framework/interface/ProductProvenanceRetriever.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include "TObjArray.h"
#include "RVersion.h"
#include "TDictAttributeMap.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include "boost/algorithm/string.hpp"

namespace {
  edm::rntuple_temp::RNTupleTempOutputModule::Optimizations fromConfig(edm::ParameterSet const& iConfig) {
    edm::rntuple_temp::RNTupleTempOutputModule::Optimizations opts;
    opts.approxZippedClusterSize = iConfig.getUntrackedParameter<unsigned long long>("approxZippedClusterSize");
    opts.maxUnzippedClusterSize = iConfig.getUntrackedParameter<unsigned long long>("maxUnzippedClusterSize");
    opts.initialUnzippedPageSize = iConfig.getUntrackedParameter<unsigned long long>("initialUnzippedPageSize");
    opts.maxUnzippedPageSize = iConfig.getUntrackedParameter<unsigned long long>("maxUnzippedPageSize");
    opts.pageBufferBudget = iConfig.getUntrackedParameter<unsigned long long>("pageBufferBudget");
    opts.useBufferedWrite = iConfig.getUntrackedParameter<bool>("useBufferedWrite");
    opts.useDirectIO = iConfig.getUntrackedParameter<bool>("useDirectIO");
    return opts;
  }
}  // namespace
namespace edm::rntuple_temp {
  RNTupleTempOutputModule::RNTupleTempOutputModule(ParameterSet const& pset)
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
        optimizations_(fromConfig(pset.getUntrackedParameterSet("rntupleWriteOptions"))),
        dropMetaData_(DropNone),
        moduleLabel_(pset.getParameter<std::string>("@module_label")),
        initializedFromInput_(false),
        outputFileCount_(0),
        inputFileCount_(0),
        branchParents_(),
        productDependencies_(),
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

    // We don't use this next parameter, but we read it anyway because it is part
    // of the configuration of this module.  An external parser creates the
    // configuration by reading this source code.
    pset.getUntrackedParameterSet("dataset");
  }

  void RNTupleTempOutputModule::beginJob() {}

  void RNTupleTempOutputModule::initialRegistry(edm::ProductRegistry const& iReg) {
    reg_ = std::make_unique<ProductRegistry>(iReg.productList());
  }

  std::string const& RNTupleTempOutputModule::currentFileName() const { return rootOutputFile_->fileName(); }

  RNTupleTempOutputModule::AuxItem::AuxItem() : basketSize_(ProductDescription::invalidBasketSize) {}

  RNTupleTempOutputModule::OutputItem::OutputItem(ProductDescription const* bd,
                                                  EDGetToken const& token,
                                                  int splitLevel,
                                                  int basketSize)
      : productDescription_(bd), token_(token), product_(nullptr), splitLevel_(splitLevel), basketSize_(basketSize) {}

  namespace {
    std::regex convertBranchExpression(std::string const& iGlobBranchExpression) {
      std::string tmp(iGlobBranchExpression);
      boost::replace_all(tmp, "*", ".*");
      boost::replace_all(tmp, "?", ".");
      return std::regex(tmp);
    }
  }  // namespace

  inline bool RNTupleTempOutputModule::SpecialSplitLevelForBranch::match(std::string const& iBranchName) const {
    return std::regex_match(iBranchName, branch_);
  }

  std::regex RNTupleTempOutputModule::SpecialSplitLevelForBranch::convert(
      std::string const& iGlobBranchExpression) const {
    return convertBranchExpression(iGlobBranchExpression);
  }

  bool RNTupleTempOutputModule::AliasForBranch::match(std::string const& iBranchName) const {
    return std::regex_match(iBranchName, branch_);
  }

  std::regex RNTupleTempOutputModule::AliasForBranch::convert(std::string const& iGlobBranchExpression) const {
    return convertBranchExpression(iGlobBranchExpression);
  }

  void RNTupleTempOutputModule::fillSelectedItemList(BranchType branchType,
                                                     std::string const& processName,
                                                     OutputItemList& outputItemList) {
    SelectedProducts const& keptVector = keptProducts()[branchType];

    // Fill outputItemList with an entry for each branch.
    for (auto const& kept : keptVector) {
      int splitLevel = ProductDescription::invalidSplitLevel;
      int basketSize = ProductDescription::invalidBasketSize;

      ProductDescription const& prod = *kept.first;
      if (branchType == InProcess && processName != prod.processName()) {
        continue;
      }
      outputItemList.emplace_back(&prod, kept.second, splitLevel, basketSize);
    }
  }

  void RNTupleTempOutputModule::beginInputFile(FileBlock const& fb) {
    if (isFileOpen()) {
      //Faster to read ChildrenBranches directly from input
      // file than to build it every event
      auto const& branchToChildMap = fb.productDependencies().childLookup();
      for (auto const& parentToChildren : branchToChildMap) {
        for (auto const& child : parentToChildren.second) {
          productDependencies_.insertChild(parentToChildren.first, child);
        }
      }
      rootOutputFile_->beginInputFile(fb, remainingEvents());
    }
  }

  void RNTupleTempOutputModule::openFile(FileBlock const& fb) {
    if (!isFileOpen()) {
      reallyOpenFile();
      beginInputFile(fb);
    }
  }

  void RNTupleTempOutputModule::respondToOpenInputFile(FileBlock const& fb) {
    if (!initializedFromInput_) {
      std::vector<std::string> const& processesWithProcessBlockProducts =
          outputProcessBlockHelper().processesWithProcessBlockProducts();
      unsigned int numberOfProcessesWithProcessBlockProducts = processesWithProcessBlockProducts.size();
      unsigned int numberOfTRNTuples = numberOfRunLumiEventProductTrees + numberOfProcessesWithProcessBlockProducts;
      selectedOutputItemList_.resize(numberOfTRNTuples);

      for (unsigned int i = InEvent; i < NumBranchTypes; ++i) {
        BranchType branchType = static_cast<BranchType>(i);
        if (branchType != InProcess) {
          std::string processName;
          OutputItemList& outputItemList = selectedOutputItemList_[branchType];
          fillSelectedItemList(branchType, processName, outputItemList);
        } else {
          // Handle output items in ProcessBlocks
          for (unsigned int k = InProcess; k < numberOfTRNTuples; ++k) {
            OutputItemList& outputItemList = selectedOutputItemList_[k];
            std::string const& processName = processesWithProcessBlockProducts[k - InProcess];
            fillSelectedItemList(branchType, processName, outputItemList);
          }
        }
      }
      initializedFromInput_ = true;
    }
    ++inputFileCount_;
    beginInputFile(fb);
  }

  void RNTupleTempOutputModule::respondToCloseInputFile(FileBlock const& fb) {
    if (rootOutputFile_)
      rootOutputFile_->respondToCloseInputFile(fb);
  }

  void RNTupleTempOutputModule::setProcessesWithSelectedMergeableRunProducts(std::set<std::string> const& processes) {
    processesWithSelectedMergeableRunProducts_.assign(processes.begin(), processes.end());
  }

  RNTupleTempOutputModule::~RNTupleTempOutputModule() {}

  void RNTupleTempOutputModule::write(EventForOutput const& e) {
    updateBranchParents(e);
    rootOutputFile_->writeOne(e);
    if (!statusFileName_.empty()) {
      std::ofstream statusFile(statusFileName_.c_str());
      statusFile << e.id() << " time: " << std::setprecision(3) << TimeOfDay() << '\n';
      statusFile.close();
    }
  }

  void RNTupleTempOutputModule::writeLuminosityBlock(LuminosityBlockForOutput const& lb) {
    rootOutputFile_->writeLuminosityBlock(lb);
  }

  void RNTupleTempOutputModule::writeRun(RunForOutput const& r) {
    if (!reg_ or (reg_->size() < r.productRegistry().size())) {
      reg_ = std::make_unique<ProductRegistry>(r.productRegistry().productList());
    }
    rootOutputFile_->writeRun(r);
  }

  void RNTupleTempOutputModule::writeProcessBlock(ProcessBlockForOutput const& pb) {
    rootOutputFile_->writeProcessBlock(pb);
  }

  void RNTupleTempOutputModule::reallyCloseFile() {
    fillDependencyGraph();
    branchParents_.clear();
    startEndFile();
    writeMetaData();

    writeParameterSetRegistry();
    writeParentageRegistry();
    productDependencies_.clear();
    finishEndFile();

    doExtrasAfterCloseFile();
  }

  // At some later date, we may move functionality from finishEndFile() to here.
  void RNTupleTempOutputModule::startEndFile() {}

  void RNTupleTempOutputModule::writeMetaData() { rootOutputFile_->writeMetaData(*reg_); }

  void RNTupleTempOutputModule::writeParameterSetRegistry() { rootOutputFile_->writeParameterSetRegistry(); }
  void RNTupleTempOutputModule::writeParentageRegistry() { rootOutputFile_->writeParentageRegistry(); }
  void RNTupleTempOutputModule::finishEndFile() {
    rootOutputFile_->finishEndFile();
    rootOutputFile_ = nullptr;
  }  // propagate_const<T> has no reset() function
  void RNTupleTempOutputModule::doExtrasAfterCloseFile() {}
  bool RNTupleTempOutputModule::isFileOpen() const { return rootOutputFile_.get() != nullptr; }
  bool RNTupleTempOutputModule::shouldWeCloseFile() const { return rootOutputFile_->shouldWeCloseFile(); }

  std::pair<std::string, std::string> RNTupleTempOutputModule::physicalAndLogicalNameForNewFile() {
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

  void RNTupleTempOutputModule::reallyOpenFile() {
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

  void RNTupleTempOutputModule::updateBranchParentsForOneBranch(ProductProvenanceRetriever const* provRetriever,
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

  void RNTupleTempOutputModule::updateBranchParents(EventForOutput const& e) {
    ProductProvenanceRetriever const* provRetriever = e.productProvenanceRetrieverPtr();
    if (producedBranches_.empty()) {
      for (auto const& prod : e.productRegistry().productList()) {
        ProductDescription const& desc = prod.second;
        if (desc.produced() && desc.branchType() == InEvent && !desc.isAlias()) {
          producedBranches_.emplace_back(desc.branchID());
        }
      }
    }
    for (auto const& bid : producedBranches_) {
      updateBranchParentsForOneBranch(provRetriever, bid);
    }
  }

  void RNTupleTempOutputModule::preActionBeforeRunEventAsync(WaitingTaskHolder iTask,
                                                             ModuleCallingContext const& iModuleCallingContext,
                                                             Principal const& iPrincipal) const noexcept {
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

  void RNTupleTempOutputModule::fillDependencyGraph() {
    for (auto const& branchParent : branchParents_) {
      BranchID const& child = branchParent.first;
      std::set<ParentageID> const& eIds = branchParent.second;
      for (auto const& eId : eIds) {
        Parentage entryDesc;
        ParentageRegistry::instance()->getMapped(eId, entryDesc);
        std::vector<BranchID> const& parents = entryDesc.parents();
        for (auto const& parent : parents) {
          productDependencies_.insertChild(parent, child);
        }
      }
    }
  }

  void RNTupleTempOutputModule::fillDescription(ParameterSetDescription& desc) {
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
    desc.addOptionalUntracked<int>("basketSize", 16384)->setComment("Default ROOT basket size in output file.");
    desc.addOptionalUntracked<int>("eventAuxiliaryBasketSize", 16384)
        ->setComment("Default ROOT basket size in output file for EventAuxiliary branch.");
    desc.addOptionalUntracked<int>("eventAutoFlushCompressedSize", 20 * 1024 * 1024)->setComment("Not used by RNTuple");
    desc.addOptionalUntracked<int>("splitLevel", 99)->setComment("Default ROOT branch split level in output file.");
    desc.addOptionalUntracked<std::string>("sortBaskets", std::string("sortbasketsbyoffset"))
        ->setComment(
            "Legal values: 'sortbasketsbyoffset', 'sortbasketsbybranch', 'sortbasketsbyentry'.\n"
            "Used by ROOT when fast copying. Affects performance.");
    desc.addOptionalUntracked<int>("treeMaxVirtualSize", -1)->setComment("Not used by RNTuple.");
    desc.addOptionalUntracked<bool>("fastCloning", false)->setComment("Not used by RNTuple");
    desc.addOptionalUntracked("mergeJob", false)->setComment("Not used by RNTuple.");
    desc.addOptionalUntracked<bool>("compactEventAuxiliary", false)->setComment("Not used by RNTuple.");
    desc.addOptionalUntracked<bool>("overrideInputFileSplitLevels", false)->setComment("Not used by RNTuple.");
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
      specialSplit.addOptionalUntracked<std::string>("branch")->setComment("Not used by RNTuple.");
      specialSplit.addOptionalUntracked<int>("splitLevel")->setComment("Not used by RNTuple.");
      desc.addVPSetUntracked("overrideBranchesSplitLevel", specialSplit, std::vector<ParameterSet>());
    }
    {
      ParameterSetDescription alias;
      alias.addOptionalUntracked<std::string>("branch")->setComment("Not used by RNTuple.");
      alias.addOptionalUntracked<std::string>("alias")->setComment("The alias to give to the TBranch");
      desc.addVPSetOptionalUntracked("branchAliases", alias, std::vector<ParameterSet>());
    }
    {
      ParameterSetDescription optimizations;

      ROOT::RNTupleWriteOptions ops;
      optimizations.addUntracked<unsigned long long>("approxZippedClusterSize", ops.GetApproxZippedClusterSize())
          ->setComment("Approximation of the target compressed cluster size");
      optimizations.addUntracked<unsigned long long>("maxUnzippedClusterSize", ops.GetMaxUnzippedClusterSize())
          ->setComment("Memory limit for committing a cluster. High compression leads to high IO buffer size.");

      optimizations.addUntracked<unsigned long long>("initialUnzippedPageSize", ops.GetInitialUnzippedPageSize())
          ->setComment("Initially, columns start with a page of this size (bytes).");
      optimizations.addUntracked<unsigned long long>("maxUnzippedPageSize", ops.GetMaxUnzippedPageSize())
          ->setComment("Pages can grow only to the given limit (bytes).");
      optimizations.addUntracked<unsigned long long>("pageBufferBudget", 0)
          ->setComment(
              "The maximum size that the sum of all page buffers used for writing into a persistent sink are allowed "
              "to "
              "use."
              " If set to zero, RNTuple will auto-adjust the budget based on the value of 'approxZippedClusterSize'."
              " If set manually, the size needs to be large enough to hold all initial page buffers.");

      optimizations.addUntracked<bool>("useBufferedWrite", ops.GetUseBufferedWrite())
          ->setComment(
              "Turn on use of buffered writing. This buffers compressed pages in memory, reorders them to keep pages "
              "of "
              "the same column adjacent, and coalesces the writes when committing a cluster.");
      optimizations.addUntracked<bool>("useDirectIO", ops.GetUseDirectIO())
          ->setComment(
              "Set use of direct IO. this introduces alignment requirements that may vary between filesystems and "
              "platforms");
      desc.addUntracked("rntupleWriteOptions", optimizations)
          ->setComment("Options to control RNTuple specific output features.");
    }
    OutputModule::fillDescription(desc);
  }

  void RNTupleTempOutputModule::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    RNTupleTempOutputModule::fillDescription(desc);
    descriptions.add("edmOutput", desc);
  }
}  // namespace edm::rntuple_temp
