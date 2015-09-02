#include "IOPool/Output/interface/PoolOutputModule.h"

#include "IOPool/Output/src/RootOutputFile.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include "TTree.h"
#include "TBranchElement.h"
#include "TObjArray.h"
#include "RVersion.h"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace edm {
  PoolOutputModule::PoolOutputModule(ParameterSet const& pset) :
  edm::one::OutputModuleBase::OutputModuleBase(pset),
  one::OutputModule<WatchInputFiles>(pset),
    rootServiceChecker_(),
    auxItems_(),
    selectedOutputItemList_(),
    fileName_(pset.getUntrackedParameter<std::string>("fileName")),
    logicalFileName_(pset.getUntrackedParameter<std::string>("logicalFileName")),
    catalog_(pset.getUntrackedParameter<std::string>("catalog")),
    maxFileSize_(pset.getUntrackedParameter<int>("maxSize")),
    compressionLevel_(pset.getUntrackedParameter<int>("compressionLevel")),
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,30,0)
    compressionAlgorithm_(pset.getUntrackedParameter<std::string>("compressionAlgorithm")),
#else
    compressionAlgorithm_("ZLIB"),
#endif
    basketSize_(pset.getUntrackedParameter<int>("basketSize")),
    eventAutoFlushSize_(pset.getUntrackedParameter<int>("eventAutoFlushCompressedSize")),
    splitLevel_(std::min<int>(pset.getUntrackedParameter<int>("splitLevel") + 1, 99)),
    basketOrder_(pset.getUntrackedParameter<std::string>("sortBaskets")),
    treeMaxVirtualSize_(pset.getUntrackedParameter<int>("treeMaxVirtualSize")),
    whyNotFastClonable_(pset.getUntrackedParameter<bool>("fastCloning") ? FileBlock::CanFastClone : FileBlock::DisabledInConfigFile),
    dropMetaData_(DropNone),
    moduleLabel_(pset.getParameter<std::string>("@module_label")),
    initializedFromInput_(false),
    outputFileCount_(0),
    inputFileCount_(0),
    childIndex_(0U),
    numberOfDigitsInIndex_(0U),
    branchParents_(),
    branchChildren_(),
    overrideInputFileSplitLevels_(pset.getUntrackedParameter<bool>("overrideInputFileSplitLevels")),
    rootOutputFile_(),
    statusFileName_() {

      if (pset.getUntrackedParameter<bool>("writeStatusFile")) {
        std::ostringstream statusfilename;
        statusfilename << moduleLabel_ << '_' << getpid();
        statusFileName_ = statusfilename.str();
      }

      std::string dropMetaData(pset.getUntrackedParameter<std::string>("dropMetaData"));
      if(dropMetaData.empty()) dropMetaData_ = DropNone;
      else if(dropMetaData == std::string("NONE")) dropMetaData_ = DropNone;
      else if(dropMetaData == std::string("DROPPED")) dropMetaData_ = DropDroppedPrior;
      else if(dropMetaData == std::string("PRIOR")) dropMetaData_ = DropPrior;
      else if(dropMetaData == std::string("ALL")) dropMetaData_ = DropAll;
      else {
        throw edm::Exception(errors::Configuration, "Illegal dropMetaData parameter value: ")
            << dropMetaData << ".\n"
            << "Legal values are 'NONE', 'DROPPED', 'PRIOR', and 'ALL'.\n";
      }

    if (!wantAllEvents()) {
      whyNotFastClonable_+= FileBlock::EventSelectionUsed;
    }

    // We don't use this next parameter, but we read it anyway because it is part
    // of the configuration of this module.  An external parser creates the
    // configuration by reading this source code.
    pset.getUntrackedParameterSet("dataset");
  }

  void PoolOutputModule::beginJob() {
    for(int i = InEvent; i < NumBranchTypes; ++i) {
      BranchType branchType = static_cast<BranchType>(i);
      SelectedProducts const& keptVector = keptProducts()[branchType];
      for(auto const& prod : keptVector) {
        checkClassDictionaries(TypeID(prod->wrappedType().typeInfo()), false);
      }
    }
  }

  std::string const& PoolOutputModule::currentFileName() const {
    return rootOutputFile_->fileName();
  }

  PoolOutputModule::AuxItem::AuxItem() :
        basketSize_(BranchDescription::invalidBasketSize) {}

  PoolOutputModule::OutputItem::OutputItem() :
        branchDescription_(0),
        product_(0),
        splitLevel_(BranchDescription::invalidSplitLevel),
        basketSize_(BranchDescription::invalidBasketSize) {}

  PoolOutputModule::OutputItem::OutputItem(BranchDescription const* bd, int splitLevel, int basketSize) :
        branchDescription_(bd),
        product_(0),
        splitLevel_(splitLevel),
        basketSize_(basketSize) {}


  PoolOutputModule::OutputItem::Sorter::Sorter(TTree* tree) : treeMap_(new std::map<std::string, int>) {
    // Fill a map mapping branch names to an index specifying the order in the tree.
    if(tree != 0) {
      TObjArray* branches = tree->GetListOfBranches();
      for(int i = 0; i < branches->GetEntries(); ++i) {
        TBranchElement* br = (TBranchElement*)branches->At(i);
        treeMap_->insert(std::make_pair(std::string(br->GetName()), i));
      }
    }
  }

  bool
  PoolOutputModule::OutputItem::Sorter::operator()(OutputItem const& lh, OutputItem const& rh) const {
    // Provides a comparison for sorting branches according to the index values in treeMap_.
    // Branches not found are always put at the end (i.e. not found > found).
    if(treeMap_->empty()) return lh < rh;
    std::string const& lstring = lh.branchDescription_->branchName();
    std::string const& rstring = rh.branchDescription_->branchName();
    std::map<std::string, int>::const_iterator lit = treeMap_->find(lstring);
    std::map<std::string, int>::const_iterator rit = treeMap_->find(rstring);
    bool lfound = (lit != treeMap_->end());
    bool rfound = (rit != treeMap_->end());
    if(lfound && rfound) {
      return lit->second < rit->second;
    } else if(lfound) {
      return true;
    } else if(rfound) {
      return false;
    }
    return lh < rh;
  }

  void PoolOutputModule::fillSelectedItemList(BranchType branchType, TTree* theInputTree) {

    SelectedProducts const& keptVector = keptProducts()[branchType];
    OutputItemList&   outputItemList = selectedOutputItemList_[branchType];
    AuxItem&   auxItem = auxItems_[branchType];

    // Fill AuxItem
    if (theInputTree != 0 && !overrideInputFileSplitLevels_) {
      TBranch* auxBranch = theInputTree->GetBranch(BranchTypeToAuxiliaryBranchName(branchType).c_str());
      if (auxBranch) {
        auxItem.basketSize_ = auxBranch->GetBasketSize();
      } else {
        auxItem.basketSize_ = basketSize_;
      }
    } else {
      auxItem.basketSize_ = basketSize_;
    }

    // Fill outputItemList with an entry for each branch.
    for(SelectedProducts::const_iterator it = keptVector.begin(), itEnd = keptVector.end(); it != itEnd; ++it) {
      int splitLevel = BranchDescription::invalidSplitLevel;
      int basketSize = BranchDescription::invalidBasketSize;

      BranchDescription const& prod = **it;
      TBranch* theBranch = ((!prod.produced() && theInputTree != 0 && !overrideInputFileSplitLevels_) ? theInputTree->GetBranch(prod.branchName().c_str()) : 0);

      if(theBranch != 0) {
        splitLevel = theBranch->GetSplitLevel();
        basketSize = theBranch->GetBasketSize();
      } else {
        splitLevel = (prod.splitLevel() == BranchDescription::invalidSplitLevel ? splitLevel_ : prod.splitLevel());
        basketSize = (prod.basketSize() == BranchDescription::invalidBasketSize ? basketSize_ : prod.basketSize());
      }
      outputItemList.emplace_back(&prod, splitLevel, basketSize);
    }

    // Sort outputItemList to allow fast copying.
    // The branches in outputItemList must be in the same order as in the input tree, with all new branches at the end.
    sort_all(outputItemList, OutputItem::Sorter(theInputTree));
  }

  void PoolOutputModule::beginInputFile(FileBlock const& fb) {
    if(isFileOpen()) {
      rootOutputFile_->beginInputFile(fb, remainingEvents());
    }
  }

  void PoolOutputModule::openFile(FileBlock const& fb) {
    if(!isFileOpen()) {
      reallyOpenFile();
      beginInputFile(fb);
    }
  }

  void PoolOutputModule::respondToOpenInputFile(FileBlock const& fb) {
    if(!initializedFromInput_) {
      for(int i = InEvent; i < NumBranchTypes; ++i) {
        BranchType branchType = static_cast<BranchType>(i);
        TTree* theInputTree = (branchType == InEvent ? fb.tree() :
                              (branchType == InLumi ? fb.lumiTree() :
                               fb.runTree()));
        fillSelectedItemList(branchType, theInputTree);
      }
      initializedFromInput_ = true;
    }
    ++inputFileCount_;
    beginInputFile(fb);
  }

  void PoolOutputModule::respondToCloseInputFile(FileBlock const& fb) {
    if(rootOutputFile_) rootOutputFile_->respondToCloseInputFile(fb);
  }

  void PoolOutputModule::postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren) {
    childIndex_ = iChildIndex;
    while (iNumberOfChildren != 0) {
      ++numberOfDigitsInIndex_;
      iNumberOfChildren /= 10;
    }
    if (numberOfDigitsInIndex_ == 0) {
      numberOfDigitsInIndex_ = 3; // Protect against zero iNumberOfChildren
    }
  }

  PoolOutputModule::~PoolOutputModule() {
  }

  void PoolOutputModule::write(EventPrincipal const& e, ModuleCallingContext const* mcc) {
    updateBranchParents(e);
    rootOutputFile_->writeOne(e, mcc);
      if (!statusFileName_.empty()) {
        std::ofstream statusFile(statusFileName_.c_str());
        statusFile << e.id() << " time: " << std::setprecision(3) << TimeOfDay() << '\n';
        statusFile.close();
      }
  }

  void PoolOutputModule::writeLuminosityBlock(LuminosityBlockPrincipal const& lb, ModuleCallingContext const* mcc) {
    rootOutputFile_->writeLuminosityBlock(lb, mcc);
  }

  void PoolOutputModule::writeRun(RunPrincipal const& r, ModuleCallingContext const* mcc) {
    rootOutputFile_->writeRun(r, mcc);
  }

  void PoolOutputModule::reallyCloseFile() {
    fillDependencyGraph();
    branchParents_.clear();
    branchChildren_.clear();
    startEndFile();
    writeFileFormatVersion();
    writeFileIdentifier();
    writeIndexIntoFile();
    writeProcessHistoryRegistry();
    writeParameterSetRegistry();
    writeProductDescriptionRegistry();
    writeParentageRegistry();
    writeBranchIDListRegistry();
    writeThinnedAssociationsHelper();
    writeProductDependencies();
    finishEndFile();

    doExtrasAfterCloseFile();
  }

  
  // At some later date, we may move functionality from finishEndFile() to here.
  void PoolOutputModule::startEndFile() { }

  void PoolOutputModule::writeFileFormatVersion() { rootOutputFile_->writeFileFormatVersion(); }
  void PoolOutputModule::writeFileIdentifier() { rootOutputFile_->writeFileIdentifier(); }
  void PoolOutputModule::writeIndexIntoFile() { rootOutputFile_->writeIndexIntoFile(); }
  void PoolOutputModule::writeProcessHistoryRegistry() { rootOutputFile_->writeProcessHistoryRegistry(); }
  void PoolOutputModule::writeParameterSetRegistry() { rootOutputFile_->writeParameterSetRegistry(); }
  void PoolOutputModule::writeProductDescriptionRegistry() { rootOutputFile_->writeProductDescriptionRegistry(); }
  void PoolOutputModule::writeParentageRegistry() { rootOutputFile_->writeParentageRegistry(); }
  void PoolOutputModule::writeBranchIDListRegistry() { rootOutputFile_->writeBranchIDListRegistry(); }
  void PoolOutputModule::writeThinnedAssociationsHelper() { rootOutputFile_->writeThinnedAssociationsHelper(); }
  void PoolOutputModule::writeProductDependencies() { rootOutputFile_->writeProductDependencies(); }
  void PoolOutputModule::finishEndFile() { rootOutputFile_->finishEndFile(); rootOutputFile_.reset(); }
  void PoolOutputModule::doExtrasAfterCloseFile() {}
  bool PoolOutputModule::isFileOpen() const { return rootOutputFile_.get() != 0; }
  bool PoolOutputModule::shouldWeCloseFile() const { return rootOutputFile_->shouldWeCloseFile(); }

  std::pair<std::string, std::string>
  PoolOutputModule::physicalAndLogicalNameForNewFile() {
      if(inputFileCount_ == 0) {
        throw edm::Exception(errors::LogicError)
          << "Attempt to open output file before input file. "
          << "Please report this to the core framework developers.\n";
      }
      std::string suffix(".root");
      std::string::size_type offset = fileName().rfind(suffix);
      bool ext = (offset == fileName().size() - suffix.size());
      if(!ext) suffix.clear();
      std::string fileBase(ext ? fileName().substr(0, offset) : fileName());
      std::ostringstream ofilename;
      std::ostringstream lfilename;
      ofilename << fileBase;
      lfilename << logicalFileName();
      if(numberOfDigitsInIndex_) {
        ofilename << '_' << std::setw(numberOfDigitsInIndex_) << std::setfill('0') << childIndex_;
        if(!logicalFileName().empty()) {
          lfilename << '_' << std::setw(numberOfDigitsInIndex_) << std::setfill('0') << childIndex_;
        }
      }
      if(outputFileCount_) {
        ofilename << std::setw(3) << std::setfill('0') << outputFileCount_;
        if(!logicalFileName().empty()) {
          lfilename << std::setw(3) << std::setfill('0') << outputFileCount_;
        }
      }
      ofilename << suffix;
      ++outputFileCount_;

      return std::make_pair(ofilename.str(), lfilename.str());
  }

  void PoolOutputModule::reallyOpenFile() {
    auto names = physicalAndLogicalNameForNewFile();
    rootOutputFile_.reset( new RootOutputFile(this, names.first, names.second));
  }

  void
  PoolOutputModule::updateBranchParents(EventPrincipal const& ep) {
    for(EventPrincipal::const_iterator i = ep.begin(), iEnd = ep.end(); i != iEnd; ++i) {
      if((*i) && (*i)->productProvenancePtr() != 0) {
        BranchID const& bid = (*i)->branchDescription().branchID();
        BranchParents::iterator it = branchParents_.find(bid);
        if(it == branchParents_.end()) {
          it = branchParents_.insert(std::make_pair(bid, std::set<ParentageID>())).first;
        }
        it->second.insert((*i)->productProvenancePtr()->parentageID());
        branchChildren_.insertEmpty(bid);
      }
    }
  }

  void
  PoolOutputModule::fillDependencyGraph() {
    for(BranchParents::const_iterator i = branchParents_.begin(), iEnd = branchParents_.end();
        i != iEnd; ++i) {
      BranchID const& child = i->first;
      std::set<ParentageID> const& eIds = i->second;
      for(std::set<ParentageID>::const_iterator it = eIds.begin(), itEnd = eIds.end();
          it != itEnd; ++it) {
        Parentage entryDesc;
        ParentageRegistry::instance()->getMapped(*it, entryDesc);
        std::vector<BranchID> const& parents = entryDesc.parents();
        for(std::vector<BranchID>::const_iterator j = parents.begin(), jEnd = parents.end();
          j != jEnd; ++j) {
          branchChildren_.insertChild(*j, child);
        }
      }
    }
  }

  void
  PoolOutputModule::fillDescription(ParameterSetDescription& desc) {
    std::string defaultString;

    desc.setComment("Writes runs, lumis, and events into EDM/ROOT files.");
    desc.addUntracked<std::string>("fileName")
        ->setComment("Name of output file.");
    desc.addUntracked<std::string>("logicalFileName", defaultString)
        ->setComment("Passed to job report. Otherwise unused by module.");
    desc.addUntracked<std::string>("catalog", defaultString)
        ->setComment("Passed to job report. Otherwise unused by module.");
    desc.addUntracked<int>("maxSize", 0x7f000000)
        ->setComment("Maximum output file size, in kB.\n"
                     "If over maximum, new output file will be started at next input file transition.");
    desc.addUntracked<int>("compressionLevel", 7)
        ->setComment("ROOT compression level of output file.");
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,30,0)
    desc.addUntracked<std::string>("compressionAlgorithm", "ZLIB")
        ->setComment("Algorithm used to compress data in the ROOT output file, allowed values are ZLIB and LZMA");
#endif
    desc.addUntracked<int>("basketSize", 16384)
        ->setComment("Default ROOT basket size in output file.");
    desc.addUntracked<int>("eventAutoFlushCompressedSize",-1)->setComment("Set ROOT auto flush stored data size (in bytes) for event TTree. The value sets how large the compressed buffer is allowed to get. The uncompressed buffer can be quite a bit larger than this depending on the average compression ratio. The value of -1 just uses ROOT's default value. The value of 0 turns off this feature.");
    desc.addUntracked<int>("splitLevel", 99)
        ->setComment("Default ROOT branch split level in output file.");
    desc.addUntracked<std::string>("sortBaskets", std::string("sortbasketsbyoffset"))
        ->setComment("Legal values: 'sortbasketsbyoffset', 'sortbasketsbybranch', 'sortbasketsbyentry'.\n"
                     "Used by ROOT when fast copying. Affects performance.");
    desc.addUntracked<int>("treeMaxVirtualSize", -1)
        ->setComment("Size of ROOT TTree TBasket cache.  Affects performance.");
    desc.addUntracked<bool>("fastCloning", true)
        ->setComment("True:  Allow fast copying, if possible.\n"
                     "False: Disable fast copying.");
    desc.addUntracked<bool>("overrideInputFileSplitLevels", false)
        ->setComment("False: Use branch split levels and basket sizes from input file, if possible.\n"
                     "True:  Always use specified or default split levels and basket sizes.");
    desc.addUntracked<bool>("writeStatusFile", false)
        ->setComment("Write a status file. Intended for use by workflow management.");
    desc.addUntracked<std::string>("dropMetaData", defaultString)
        ->setComment("Determines handling of per product per event metadata.  Options are:\n"
                     "'NONE':    Keep all of it.\n"
                     "'DROPPED': Keep it for products produced in current process and all kept products. Drop it for dropped products produced in prior processes.\n"
                     "'PRIOR':   Keep it for products produced in current process. Drop it for products produced in prior processes.\n"
                     "'ALL':     Drop all of it.");
    ParameterSetDescription dataSet;
    dataSet.setAllowAnything();
    desc.addUntracked<ParameterSetDescription>("dataset", dataSet)
     ->setComment("PSet is only used by Data Operations and not by this module.");

    OutputModule::fillDescription(desc);
  }

  void
  PoolOutputModule::fillDescriptions(ConfigurationDescriptions & descriptions) {
    ParameterSetDescription desc;
    PoolOutputModule::fillDescription(desc);
    descriptions.add("edmOutput", desc);
  }
}
