
#include "FWIO/RNTupleTempOutput/src/RootOutputFile.h"

#include "FWCore/Utilities/interface/GlobalIdentifier.h"

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "FWCore/Version/interface/GetFileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Common/interface/OutputProcessBlockHelper.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/MergeableRunProductMetadata.h"
#include "FWCore/Framework/interface/OccurrenceForOutput.h"
#include "FWCore/Framework/interface/ProcessBlockForOutput.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Provenance/interface/ProductDependencies.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/EventToProcessBlockIndexes.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/StoredProcessBlockHelper.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/ExceptionPropagate.h"
#include "IOPool/Common/interface/getWrapperBasePtr.h"
#include "IOPool/Provenance/interface/CommonProvenanceFiller.h"

#include "TFile.h"
#include "TClass.h"
#include "Rtypes.h"
#include "RVersion.h"

#include "ROOT/RNTuple.hxx"
#include "ROOT/RNTupleWriter.hxx"

#include "Compression.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace edm::rntuple_temp {

  namespace {
    bool sorterForJobReportHash(ProductDescription const* lh, ProductDescription const* rh) {
      return lh->fullClassName() < rh->fullClassName()               ? true
             : lh->fullClassName() > rh->fullClassName()             ? false
             : lh->moduleLabel() < rh->moduleLabel()                 ? true
             : lh->moduleLabel() > rh->moduleLabel()                 ? false
             : lh->productInstanceName() < rh->productInstanceName() ? true
             : lh->productInstanceName() > rh->productInstanceName() ? false
             : lh->processName() < rh->processName()                 ? true
                                                                     : false;
    }

    TFile* openTFile(char const* name, int compressionLevel) {
      TFile* file = TFile::Open(name, "recreate", "", compressionLevel);
      std::exception_ptr e = edm::threadLocalException::getException();
      if (e != std::exception_ptr()) {
        edm::threadLocalException::setException(std::exception_ptr());
        std::rethrow_exception(e);
      }
      return file;
    }

    std::string fixName(std::string_view iName) {
      if (not iName.empty() and iName.back() == '.') {
        iName.remove_suffix(1);
      }
      return std::string(iName);
    }

  }  // namespace

  RootOutputFile::RootOutputFile(RNTupleTempOutputModule* om,
                                 std::string const& fileName,
                                 std::string const& logicalFileName,
                                 std::vector<std::string> const& processesWithSelectedMergeableRunProducts,
                                 std::string const& overrideGUID)
      : file_(fileName),
        logicalFile_(logicalFileName),
        reportToken_(0),
        om_(om),
        filePtr_(openTFile(file_.c_str(), om_->compressionLevel())),
        fid_(),
        eventEntryNumber_(0LL),
        lumiEntryNumber_(0LL),
        runEntryNumber_(0LL),
        indexIntoFile_(),
        storedMergeableRunProductMetadata_(processesWithSelectedMergeableRunProducts),
        nEventsInLumi_(0),
        lumiAux_(),
        runAux_(),
        pEventAux_(nullptr),
        pLumiAux_(&lumiAux_),
        pRunAux_(&runAux_),
        eventEntryInfoVector_(),
        pEventEntryInfoVector_(&eventEntryInfoVector_),
        pBranchListIndexes_(nullptr),
        pEventSelectionIDs_(nullptr),
        eventRNTuple_(filePtr(), InEvent),
        lumiRNTuple_(filePtr(), InLumi),
        runRNTuple_(filePtr(), InRun),
        dataTypeReported_(false),
        processHistoryRegistry_(),
        parentageIDs_(),
        branchesWithStoredHistory_(),
        wrapperBaseTClass_(TClass::GetClass("edm::WrapperBase")) {
    std::vector<std::string> const& processesWithProcessBlockProducts =
        om_->outputProcessBlockHelper().processesWithProcessBlockProducts();
    for (auto const& processName : processesWithProcessBlockProducts) {
      processBlockRNTuples_.emplace_back(std::make_unique<RootOutputRNTuple>(filePtr(), InProcess, processName));
    }

    if (om_->compressionAlgorithm() == std::string("ZLIB")) {
      filePtr_->SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kZLIB);
    } else if (om_->compressionAlgorithm() == std::string("LZMA")) {
      filePtr_->SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kLZMA);
    } else if (om_->compressionAlgorithm() == std::string("ZSTD")) {
      filePtr_->SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kZSTD);
    } else if (om_->compressionAlgorithm() == std::string("LZ4")) {
      filePtr_->SetCompressionAlgorithm(ROOT::RCompressionSetting::EAlgorithm::kLZ4);
    } else {
      throw Exception(errors::Configuration)
          << "RNTupleTempOutputModule configured with unknown compression algorithm '" << om_->compressionAlgorithm()
          << "'\n"
          << "Allowed compression algorithms are ZLIB, LZMA, LZ4, and ZSTD\n";
    }
    eventRNTuple_.addAuxiliary<EventAuxiliary>(
        BranchTypeToAuxiliaryBranchName(InEvent), &pEventAux_, om_->auxItems()[InEvent].basketSize_);

    eventRNTuple_.addAuxiliary<StoredProductProvenanceVector>(BranchTypeToProductProvenanceBranchName(InEvent),
                                                              &pEventEntryInfoVector_,
                                                              om_->auxItems()[InEvent].basketSize_);
    eventRNTuple_.addAuxiliary<EventSelectionIDVector>(
        poolNames::eventSelectionsBranchName(), &pEventSelectionIDs_, om_->auxItems()[InEvent].basketSize_);
    eventRNTuple_.addAuxiliary<BranchListIndexes>(
        poolNames::branchListIndexesBranchName(), &pBranchListIndexes_, om_->auxItems()[InEvent].basketSize_);

    if (om_->outputProcessBlockHelper().productsFromInputKept()) {
      eventRNTuple_.addAuxiliary<EventToProcessBlockIndexes>(poolNames::eventToProcessBlockIndexesBranchName(),
                                                             &pEventToProcessBlockIndexes_,
                                                             om_->auxItems()[InEvent].basketSize_);
    }

    lumiRNTuple_.addAuxiliary<LuminosityBlockAuxiliary>(
        BranchTypeToAuxiliaryBranchName(InLumi), &pLumiAux_, om_->auxItems()[InLumi].basketSize_);

    runRNTuple_.addAuxiliary<RunAuxiliary>(
        BranchTypeToAuxiliaryBranchName(InRun), &pRunAux_, om_->auxItems()[InRun].basketSize_);

    treePointers_.emplace_back(&eventRNTuple_);
    treePointers_.emplace_back(&lumiRNTuple_);
    treePointers_.emplace_back(&runRNTuple_);
    for (auto& processBlockRNTuple : processBlockRNTuples_) {
      treePointers_.emplace_back(processBlockRNTuple.get());
    }

    for (unsigned int i = 0; i < treePointers_.size(); ++i) {
      RootOutputRNTuple* theRNTuple = treePointers_[i];
      for (auto& item : om_->selectedOutputItemList()[i]) {
        item.setProduct(nullptr);
        ProductDescription const& desc = *item.productDescription();
        theRNTuple->addField(fixName(desc.branchName()),
                             desc.wrappedName(),
                             item.productPtr(),
                             item.splitLevel(),
                             item.basketSize(),
                             item.productDescription()->produced());
        //make sure we always store product registry info for all branches we create
        branchesWithStoredHistory_.insert(item.branchID());
      }
    }
    RootOutputRNTuple::Config config;
    config.compressionAlgo =
        om_->compressionAlgorithm() == std::string("LZMA")   ? RootOutputRNTuple::Config::CompressionAlgos::kLZMA
        : om_->compressionAlgorithm() == std::string("ZSTD") ? RootOutputRNTuple::Config::CompressionAlgos::kZSTD
        : om_->compressionAlgorithm() == std::string("LZ4")  ? RootOutputRNTuple::Config::CompressionAlgos::kLZ4
                                                             : RootOutputRNTuple::Config::CompressionAlgos::kZLIB;
    config.compressionLevel = om_->compressionLevel();
    auto& optimizations = om_->optimizations();
    config.approxZippedClusterSize = optimizations.approxZippedClusterSize;
    config.maxUnzippedClusterSize = optimizations.maxUnzippedClusterSize;
    config.initialUnzippedPageSize = optimizations.initialUnzippedPageSize;
    config.maxUnzippedPageSize = optimizations.maxUnzippedPageSize;
    config.pageBufferBudget = optimizations.pageBufferBudget;
    config.useBufferedWrite = optimizations.useBufferedWrite;
    config.useDirectIO = optimizations.useDirectIO;
    for (auto& tree : treePointers_) {
      tree->finishInitialization(config);
    }

    if (overrideGUID.empty()) {
      fid_ = FileID(createGlobalIdentifier());
    } else {
      if (not isValidGlobalIdentifier(overrideGUID)) {
        throw edm::Exception(errors::Configuration)
            << "GUID to be used for output file is not valid (is '" << overrideGUID << "')";
      }
      fid_ = FileID(overrideGUID);
    }

    // For the Job Report, get a vector of branch names in the "Events" tree.
    // Also create a hash of all the branch names in the "Events" tree
    // in a deterministic order, except use the full class name instead of the friendly class name.
    // To avoid extra string copies, we create a vector of pointers into the product registry,
    // and use a custom comparison operator for sorting.
    std::vector<std::string> branchNames;
    std::vector<ProductDescription const*> branches;
    branchNames.reserve(om_->selectedOutputItemList()[InEvent].size());
    branches.reserve(om->selectedOutputItemList()[InEvent].size());
    for (auto const& item : om_->selectedOutputItemList()[InEvent]) {
      branchNames.push_back(item.productDescription()->branchName());
      branches.push_back(item.productDescription());
    }
    // Now sort the branches for the hash.
    sort_all(branches, sorterForJobReportHash);
    // Now, make a concatenated string.
    std::ostringstream oss;
    char const underscore = '_';
    for (auto const& branch : branches) {
      ProductDescription const& bd = *branch;
      oss << bd.fullClassName() << underscore << bd.moduleLabel() << underscore << bd.productInstanceName()
          << underscore << bd.processName() << underscore;
    }
    std::string stringrep = oss.str();
    cms::Digest md5alg(stringrep);

    // Register the output file with the JobReport service
    // and get back the token for it.
    std::string moduleName = "RNTupleTempOutputModule";
    Service<JobReport> reportSvc;
    reportToken_ = reportSvc->outputFileOpened(file_,
                                               logicalFile_,        // PFN and LFN
                                               om_->catalog(),      // catalog
                                               moduleName,          // module class name
                                               om_->moduleLabel(),  // module label
                                               fid_.fid(),          // file id (guid)
                                               std::string(),       // data type (not yet known, so string is empty).
                                               md5alg.digest().toString(),  // branch hash
                                               branchNames);                // branch names being written
  }

  void RootOutputFile::beginInputFile(FileBlock const& fb, int remainingEvents) {}

  void RootOutputFile::respondToCloseInputFile(FileBlock const&) {
    // We can't do setEntries() on the event tree if the EventAuxiliary branch is empty & disabled
    eventRNTuple_.setEntries();
    lumiRNTuple_.setEntries();
    runRNTuple_.setEntries();
  }

  bool RootOutputFile::shouldWeCloseFile() const {
    unsigned int const oneK = 1024;
    Long64_t size = filePtr_->GetSize() / oneK;
    return (size >= om_->maxFileSize());
  }

  void RootOutputFile::writeOne(EventForOutput const& e) {
    // Auxiliary branch
    pEventAux_ = &e.eventAuxiliary();

    // Because getting the data may cause an exception to be thrown we want to do that
    // first before writing anything to the file about this event
    // NOTE: pEventAux_, pBranchListIndexes_, pEventSelectionIDs_, and pEventEntryInfoVector_
    // must be set before calling fillBranches since they get written out in that routine.
    assert(pEventAux_->processHistoryID() == e.processHistoryID());
    pBranchListIndexes_ = &e.branchListIndexes();
    pEventToProcessBlockIndexes_ = &e.eventToProcessBlockIndexes();

    // Note: The EventSelectionIDVector should have a one to one correspondence with the processes in the process history.
    // Therefore, a new entry should be added if and only if the current process has been added to the process history,
    // which is done if and only if there is a produced product.
    EventSelectionIDVector esids = e.eventSelectionIDs();
    if (e.productRegistry().anyProductProduced() || !om_->wantAllEvents()) {
      esids.push_back(om_->selectorConfig());
    }
    pEventSelectionIDs_ = &esids;
    ProductProvenanceRetriever const* provRetriever = e.productProvenanceRetrieverPtr();
    assert(provRetriever);
    unsigned int ttreeIndex = InEvent;
    fillBranches(InEvent, e, ttreeIndex, &eventEntryInfoVector_, provRetriever);

    // Add the dataType to the job report if it hasn't already been done
    if (!dataTypeReported_) {
      Service<JobReport> reportSvc;
      std::string dataType("MC");
      if (pEventAux_->isRealData())
        dataType = "Data";
      reportSvc->reportDataType(reportToken_, dataType);
      dataTypeReported_ = true;
    }

    // Store the process history.
    processHistoryRegistry_.registerProcessHistory(e.processHistory());
    // Store the reduced ID in the IndexIntoFile
    ProcessHistoryID reducedPHID = processHistoryRegistry_.reducedProcessHistoryID(e.processHistoryID());
    // Add event to index
    indexIntoFile_.addEntry(
        reducedPHID, pEventAux_->run(), pEventAux_->luminosityBlock(), pEventAux_->event(), eventEntryNumber_);
    ++eventEntryNumber_;

    // Report event written
    Service<JobReport> reportSvc;
    reportSvc->eventWrittenToFile(reportToken_, e.id().run(), e.id().event());
    ++nEventsInLumi_;
  }

  void RootOutputFile::writeLuminosityBlock(LuminosityBlockForOutput const& lb) {
    // Auxiliary branch
    // NOTE: lumiAux_ must be filled before calling fillBranches since it gets written out in that routine.
    lumiAux_ = lb.luminosityBlockAuxiliary();
    // Use the updated process historyID
    lumiAux_.setProcessHistoryID(lb.processHistoryID());
    // Store the process history.
    processHistoryRegistry_.registerProcessHistory(lb.processHistory());
    // Store the reduced ID in the IndexIntoFile
    ProcessHistoryID reducedPHID = processHistoryRegistry_.reducedProcessHistoryID(lb.processHistoryID());
    // Add lumi to index.
    indexIntoFile_.addEntry(reducedPHID, lumiAux_.run(), lumiAux_.luminosityBlock(), 0U, lumiEntryNumber_);
    ++lumiEntryNumber_;
    unsigned int ttreeIndex = InLumi;
    fillBranches(InLumi, lb, ttreeIndex);
    lumiRNTuple_.optimizeBaskets(10ULL * 1024 * 1024);

    Service<JobReport> reportSvc;
    reportSvc->reportLumiSection(reportToken_, lb.id().run(), lb.id().luminosityBlock(), nEventsInLumi_);
    nEventsInLumi_ = 0;
  }

  void RootOutputFile::writeRun(RunForOutput const& r) {
    // Auxiliary branch
    // NOTE: runAux_ must be filled before calling fillBranches since it gets written out in that routine.
    runAux_ = r.runAuxiliary();
    // Use the updated process historyID
    runAux_.setProcessHistoryID(r.processHistoryID());
    // Store the process history.
    processHistoryRegistry_.registerProcessHistory(r.processHistory());
    // Store the reduced ID in the IndexIntoFile
    ProcessHistoryID reducedPHID = processHistoryRegistry_.reducedProcessHistoryID(r.processHistoryID());
    // Add run to index.
    indexIntoFile_.addEntry(reducedPHID, runAux_.run(), 0U, 0U, runEntryNumber_);
    r.mergeableRunProductMetadata()->addEntryToStoredMetadata(storedMergeableRunProductMetadata_);
    ++runEntryNumber_;
    unsigned int ttreeIndex = InRun;
    fillBranches(InRun, r, ttreeIndex);
    runRNTuple_.optimizeBaskets(10ULL * 1024 * 1024);

    Service<JobReport> reportSvc;
    reportSvc->reportRunNumber(reportToken_, r.run());
  }

  void RootOutputFile::writeProcessBlock(ProcessBlockForOutput const& pb) {
    std::string const& processName = pb.processName();
    std::vector<std::string> const& processesWithProcessBlockProducts =
        om_->outputProcessBlockHelper().processesWithProcessBlockProducts();
    std::vector<std::string>::const_iterator it =
        std::find(processesWithProcessBlockProducts.cbegin(), processesWithProcessBlockProducts.cend(), processName);
    if (it == processesWithProcessBlockProducts.cend()) {
      return;
    }
    unsigned int ttreeIndex = InProcess + std::distance(processesWithProcessBlockProducts.cbegin(), it);
    fillBranches(InProcess, pb, ttreeIndex);
    treePointers_[ttreeIndex]->optimizeBaskets(10ULL * 1024 * 1024);
  }

  void RootOutputFile::writeMetaData(ProductRegistry const& iReg) {
    auto model = ROOT::RNTupleModel::CreateBare();
    {
      model->AddField(setupFileFormatVersion());
      model->AddField(setupFileIdentifier());
      model->AddField(setupIndexIntoFile());
      model->AddField(setupStoredMergeableRunProductMetadata());
      model->AddField(setupProcessHistoryRegistry());
      model->AddField(setupProductDescriptionRegistry());
      model->AddField(setupBranchIDListRegistry());
      model->AddField(setupThinnedAssociationsHelper());
      model->AddField(setupProductDependencies());
      if (!om_->outputProcessBlockHelper().processesWithProcessBlockProducts().empty()) {
        model->AddField(setupProcessBlockHelper());
      }
    }

    auto writeOptions = ROOT::RNTupleWriteOptions();
    //writeOptions.SetCompression(convert(iConfig.compressionAlgo), iConfig.compressionLevel);
    auto metaData =
        ROOT::RNTupleWriter::Append(std::move(model), poolNames::metaDataTreeName(), *filePtr_, writeOptions);

    auto rentry = metaData->CreateEntry();

    writeFileFormatVersion(*rentry);
    writeFileIdentifier(*rentry);
    writeIndexIntoFile(*rentry);
    writeStoredMergeableRunProductMetadata(*rentry);
    writeProcessHistoryRegistry(*rentry);
    writeProductDescriptionRegistry(*rentry, iReg);
    writeBranchIDListRegistry(*rentry);
    writeThinnedAssociationsHelper(*rentry);
    writeProductDependencies(*rentry);
    writeProcessBlockHelper(*rentry);
    metaData->Fill(*rentry);
  }

  void RootOutputFile::writeParentageRegistry() {
    auto model = ROOT::RNTupleModel::CreateBare();
    model->AddField(ROOT::RFieldBase::Create(poolNames::parentageBranchName(), "edm::Parentage").Unwrap());

    auto writeOptions = ROOT::RNTupleWriteOptions();
    //writeOptions.SetCompression(convert(iConfig.compressionAlgo), iConfig.compressionLevel);
    auto parentageWriter =
        ROOT::RNTupleWriter::Append(std::move(model), poolNames::parentageTreeName(), *filePtr_, writeOptions);

    ParentageRegistry& ptReg = *ParentageRegistry::instance();

    std::vector<ParentageID> orderedIDs(parentageIDs_.size());
    for (auto const& parentageID : parentageIDs_) {
      orderedIDs[parentageID.second] = parentageID.first;
    }
    auto rentry = parentageWriter->CreateEntry();

    //now put them into the RNTuple in the correct order
    for (auto const& orderedID : orderedIDs) {
      rentry->BindRawPtr(poolNames::parentageBranchName(), const_cast<edm::Parentage*>(ptReg.getMapped(orderedID)));
      parentageWriter->Fill(*rentry);
    }
  }
  /////////

  std::unique_ptr<ROOT::RFieldBase> RootOutputFile::setupFileFormatVersion() {
    return ROOT::RFieldBase::Create(poolNames::fileFormatVersionBranchName(), "edm::FileFormatVersion").Unwrap();
  }

  std::unique_ptr<ROOT::RFieldBase> RootOutputFile::setupFileIdentifier() {
    return ROOT::RFieldBase::Create(poolNames::fileIdentifierBranchName(), "edm::FileID").Unwrap();
  }

  std::unique_ptr<ROOT::RFieldBase> RootOutputFile::setupIndexIntoFile() {
    return ROOT::RFieldBase::Create(poolNames::indexIntoFileBranchName(), "edm::IndexIntoFile").Unwrap();
  }

  std::unique_ptr<ROOT::RFieldBase> RootOutputFile::setupStoredMergeableRunProductMetadata() {
    return ROOT::RFieldBase::Create(poolNames::mergeableRunProductMetadataBranchName(),
                                    "edm::StoredMergeableRunProductMetadata")
        .Unwrap();
  }

  std::unique_ptr<ROOT::RFieldBase> RootOutputFile::setupProcessHistoryRegistry() {
    return ROOT::RFieldBase::Create(poolNames::processHistoryBranchName(), "std::vector<edm::ProcessHistory>").Unwrap();
  }

  std::unique_ptr<ROOT::RFieldBase> RootOutputFile::setupBranchIDListRegistry() {
    return ROOT::RFieldBase::Create(poolNames::branchIDListBranchName(), "std::vector<std::vector<unsigned int>>")
        .Unwrap();
  }

  std::unique_ptr<ROOT::RFieldBase> RootOutputFile::setupThinnedAssociationsHelper() {
    return ROOT::RFieldBase::Create(poolNames::thinnedAssociationsHelperBranchName(), "edm::ThinnedAssociationsHelper")
        .Unwrap();
  }

  std::unique_ptr<ROOT::RFieldBase> RootOutputFile::setupProductDescriptionRegistry() {
    return ROOT::RFieldBase::Create(poolNames::productDescriptionBranchName(), "edm::ProductRegistry").Unwrap();
  }
  std::unique_ptr<ROOT::RFieldBase> RootOutputFile::setupProductDependencies() {
    return ROOT::RFieldBase::Create(poolNames::productDependenciesBranchName(), "edm::ProductDependencies").Unwrap();
  }

  std::unique_ptr<ROOT::RFieldBase> RootOutputFile::setupProcessBlockHelper() {
    return ROOT::RFieldBase::Create(poolNames::processBlockHelperBranchName(), "edm::StoredProcessBlockHelper").Unwrap();
  }

  ////////
  void RootOutputFile::writeFileFormatVersion(ROOT::REntry& rentry) {
    auto fileFormatVersion = std::make_shared<FileFormatVersion>(getFileFormatVersion());
    rentry.BindValue(poolNames::fileFormatVersionBranchName(), fileFormatVersion);
  }

  void RootOutputFile::writeFileIdentifier(ROOT::REntry& rentry) {
    rentry.BindRawPtr(poolNames::fileIdentifierBranchName(), &fid_);
  }

  void RootOutputFile::writeIndexIntoFile(ROOT::REntry& rentry) {
    indexIntoFile_.sortVector_Run_Or_Lumi_Entries();
    rentry.BindRawPtr(poolNames::indexIntoFileBranchName(), &indexIntoFile_);
  }

  void RootOutputFile::writeStoredMergeableRunProductMetadata(ROOT::REntry& rentry) {
    storedMergeableRunProductMetadata_.optimizeBeforeWrite();
    rentry.BindRawPtr(poolNames::mergeableRunProductMetadataBranchName(), &storedMergeableRunProductMetadata_);
  }

  void RootOutputFile::writeProcessHistoryRegistry(ROOT::REntry& rentry) {
    auto procHistoryVector = std::make_shared<ProcessHistoryVector>();
    for (auto const& ph : processHistoryRegistry_) {
      procHistoryVector->push_back(ph.second);
    }
    rentry.BindValue(poolNames::processHistoryBranchName(), procHistoryVector);
  }

  void RootOutputFile::writeBranchIDListRegistry(ROOT::REntry& rentry) {
    rentry.BindRawPtr(poolNames::branchIDListBranchName(), const_cast<BranchIDLists*>(om_->branchIDLists()));
  }

  void RootOutputFile::writeThinnedAssociationsHelper(ROOT::REntry& rentry) {
    auto* p = const_cast<ThinnedAssociationsHelper*>(om_->thinnedAssociationsHelper());
    rentry.BindRawPtr(poolNames::thinnedAssociationsHelperBranchName(), p);
  }

  void RootOutputFile::writeProductDescriptionRegistry(ROOT::REntry& rentry, ProductRegistry const& iReg) {
    // Make a local copy of the ProductRegistry, removing any transient or pruned products.
    using ProductList = ProductRegistry::ProductList;
    auto pReg = std::make_shared<ProductRegistry>(iReg.productList());
    ProductList& pList = const_cast<ProductList&>(pReg->productList());
    for (auto const& prod : pList) {
      if (prod.second.branchID() != prod.second.originalBranchID()) {
        if (branchesWithStoredHistory_.find(prod.second.branchID()) != branchesWithStoredHistory_.end()) {
          branchesWithStoredHistory_.insert(prod.second.originalBranchID());
        }
      }
    }
    std::set<BranchID>::iterator end = branchesWithStoredHistory_.end();
    for (ProductList::iterator it = pList.begin(); it != pList.end();) {
      if (branchesWithStoredHistory_.find(it->second.branchID()) == end) {
        // avoid invalidating iterator on deletion
        ProductList::iterator itCopy = it;
        ++it;
        pList.erase(itCopy);

      } else {
        ++it;
      }
    }

    rentry.BindValue(poolNames::productDescriptionBranchName(), pReg);
  }
  void RootOutputFile::writeProductDependencies(ROOT::REntry& rentry) {
    ProductDependencies& pDeps = const_cast<ProductDependencies&>(om_->productDependencies());
    rentry.BindRawPtr(poolNames::productDependenciesBranchName(), &pDeps);
  }

  void RootOutputFile::writeProcessBlockHelper(ROOT::REntry& rentry) {
    if (!om_->outputProcessBlockHelper().processesWithProcessBlockProducts().empty()) {
      auto storedProcessBlockHelper = std::make_shared<StoredProcessBlockHelper>(
          om_->outputProcessBlockHelper().processesWithProcessBlockProducts());
      om_->outputProcessBlockHelper().fillCacheIndices(*storedProcessBlockHelper);

      rentry.BindValue(poolNames::processBlockHelperBranchName(), storedProcessBlockHelper);
    }
  }
  void RootOutputFile::writeParameterSetRegistry() {
    auto model = ROOT::RNTupleModel::CreateBare();
    {
      auto field =
          ROOT::RFieldBase::Create("IdToParameterSetsBlobs", "std::pair<edm::Hash<1>,edm::ParameterSetBlob>").Unwrap();
      model->AddField(std::move(field));
    }
    auto writeOptions = ROOT::RNTupleWriteOptions();
    //writeOptions.SetCompression(convert(iConfig.compressionAlgo), iConfig.compressionLevel);
    auto parameterSets =
        ROOT::RNTupleWriter::Append(std::move(model), poolNames::parameterSetsTreeName(), *filePtr_, writeOptions);

    std::pair<ParameterSetID, ParameterSetBlob> idToBlob;

    auto rentry = parameterSets->CreateEntry();
    rentry->BindRawPtr("IdToParameterSetsBlobs", &idToBlob);

    for (auto const& pset : *pset::Registry::instance()) {
      idToBlob.first = pset.first;
      idToBlob.second.pset() = pset.second.toString();

      parameterSets->Fill(*rentry);
    }
  }

  void RootOutputFile::finishEndFile() {
    std::string_view status = "beginning";
    std::string_view value = "";
    try {
      // close the file -- mfp
      // Just to play it safe, zero all pointers to objects in the TFile to be closed.
      status = "closing RNTuples";
      value = "";
      for (auto& treePointer : treePointers_) {
        treePointer->close();
        treePointer = nullptr;
      }
      status = "closing TFile";
      filePtr_->Close();
      filePtr_ = nullptr;  // propagate_const<T> has no reset() function

      // report that file has been closed
      status = "reporting to JobReport";
      Service<JobReport> reportSvc;
      reportSvc->outputFileClosed(reportToken_);
    } catch (cms::Exception& e) {
      e.addContext("Calling RootOutputFile::finishEndFile() while closing " + file_);
      e.addAdditionalInfo("While calling " + std::string(status) + std::string(value));
      throw;
    }
  }

  void RootOutputFile::insertAncestors(ProductProvenance const& iGetParents,
                                       ProductProvenanceRetriever const* iMapper,
                                       bool produced,
                                       std::set<BranchID> const& iProducedIDs,
                                       std::set<StoredProductProvenance>& oToFill) {
    assert(om_->dropMetaData() != RNTupleTempOutputModule::DropAll);
    assert(produced || om_->dropMetaData() != RNTupleTempOutputModule::DropPrior);
    if (om_->dropMetaData() == RNTupleTempOutputModule::DropDroppedPrior && !produced)
      return;
    std::vector<BranchID> const& parentIDs = iGetParents.parentage().parents();
    for (auto const& parentID : parentIDs) {
      branchesWithStoredHistory_.insert(parentID);
      ProductProvenance const* info = iMapper->branchIDToProvenance(parentID);
      if (info) {
        if (om_->dropMetaData() == RNTupleTempOutputModule::DropNone ||
            (iProducedIDs.end() != iProducedIDs.find(info->branchID()))) {
          if (insertProductProvenance(*info, oToFill)) {
            //haven't seen this one yet
            insertAncestors(*info, iMapper, produced, iProducedIDs, oToFill);
          }
        }
      }
    }
  }

  void RootOutputFile::fillBranches(BranchType const& branchType,
                                    OccurrenceForOutput const& occurrence,
                                    unsigned int ttreeIndex,
                                    StoredProductProvenanceVector* productProvenanceVecPtr,
                                    ProductProvenanceRetriever const* provRetriever) {
    std::vector<std::unique_ptr<WrapperBase> > dummies;

    OutputItemList& items = om_->selectedOutputItemList()[ttreeIndex];

    bool const doProvenance =
        (productProvenanceVecPtr != nullptr) && (om_->dropMetaData() != RNTupleTempOutputModule::DropAll);
    bool const keepProvenanceForPrior = doProvenance && om_->dropMetaData() != RNTupleTempOutputModule::DropPrior;

    std::set<StoredProductProvenance> provenanceToKeep;
    //
    //If we are dropping some of the meta data we need to know
    // which BranchIDs were produced in this process because
    // we may be storing meta data for only those products
    // We do this only for event products.
    std::set<BranchID> producedBranches;
    if (doProvenance && branchType == InEvent && om_->dropMetaData() != RNTupleTempOutputModule::DropNone) {
      for (auto bd : occurrence.productRegistry().allProductDescriptions()) {
        if (bd->produced() && bd->branchType() == InEvent) {
          producedBranches.insert(bd->branchID());
        }
      }
    }

    // Loop over EDProduct branches, possibly fill the provenance, and write the branch.
    for (auto& item : items) {
      BranchID const& id = item.productDescription()->branchID();
      branchesWithStoredHistory_.insert(id);

      bool produced = item.productDescription()->produced();
      bool getProd = true;
      bool keepProvenance = doProvenance && (produced || keepProvenanceForPrior);

      WrapperBase const* product = nullptr;
      ProductProvenance const* productProvenance = nullptr;
      if (getProd) {
        BasicHandle result = occurrence.getByToken(item.token(), item.productDescription()->unwrappedTypeID());
        product = result.wrapper();
        if (result.isValid() && keepProvenance) {
          productProvenance = result.provenance()->productProvenance();
        }
        if (product == nullptr) {
          // No product with this ID is in the event.
          // Add a null product.
          TClass* cp = item.productDescription()->wrappedType().getClass();
          assert(cp != nullptr);
          int offset = cp->GetBaseClassOffset(wrapperBaseTClass_);
          void* p = cp->New();
          std::unique_ptr<WrapperBase> dummy = getWrapperBasePtr(p, offset);
          product = dummy.get();
          dummies.emplace_back(std::move(dummy));
        }
        item.setProduct(product);
      }
      if (keepProvenance && productProvenance == nullptr) {
        productProvenance = provRetriever->branchIDToProvenance(item.productDescription()->originalBranchID());
      }
      if (productProvenance) {
        insertProductProvenance(*productProvenance, provenanceToKeep);
        insertAncestors(*productProvenance, provRetriever, produced, producedBranches, provenanceToKeep);
      }
    }

    if (doProvenance)
      productProvenanceVecPtr->assign(provenanceToKeep.begin(), provenanceToKeep.end());
    treePointers_[ttreeIndex]->fill();
    if (doProvenance)
      productProvenanceVecPtr->clear();
  }

  bool RootOutputFile::insertProductProvenance(const edm::ProductProvenance& iProv,
                                               std::set<edm::StoredProductProvenance>& oToInsert) {
    StoredProductProvenance toStore;
    toStore.branchID_ = iProv.branchID().id();
    std::set<edm::StoredProductProvenance>::iterator itFound = oToInsert.find(toStore);
    if (itFound == oToInsert.end()) {
      //get the index to the ParentageID or insert a new value if not already present
      std::pair<std::map<edm::ParentageID, unsigned int>::iterator, bool> i =
          parentageIDs_.insert(std::make_pair(iProv.parentageID(), static_cast<unsigned int>(parentageIDs_.size())));
      toStore.parentageIDIndex_ = i.first->second;
      if (toStore.parentageIDIndex_ >= parentageIDs_.size()) {
        throw edm::Exception(errors::LogicError)
            << "RootOutputFile::insertProductProvenance\n"
            << "The parentage ID index value " << toStore.parentageIDIndex_
            << " is out of bounds.  The maximum value is currently " << parentageIDs_.size() - 1 << ".\n"
            << "This should never happen.\n"
            << "Please report this to the framework developers.";
      }

      oToInsert.insert(toStore);
      return true;
    }
    return false;
  }
}  // namespace edm::rntuple_temp
