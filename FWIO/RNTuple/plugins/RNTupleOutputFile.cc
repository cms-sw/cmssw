#include "RNTupleOutputFile.h"

#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/ProductProvenanceRetriever.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Utilities/interface/ConvertException.h"

#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/ProductDependencies.h"

#include "IOPool/Common/interface/getWrapperBasePtr.h"

#include "TFile.h"
#include "ROOT/RNTuple.hxx"
#include "ROOT/RNTupleWriter.hxx"
#include <string>
#include <optional>
#include <map>

namespace {
  ROOT::RCompressionSetting::EAlgorithm::EValues convert(edm::rntuple::CompressionAlgos iAlgos) {
    using namespace edm::rntuple;
    using namespace ROOT;
    switch (iAlgos) {
      case CompressionAlgos::kLZMA:
        return RCompressionSetting::EAlgorithm::kLZMA;
      case CompressionAlgos::kZSTD:
        return RCompressionSetting::EAlgorithm::kZSTD;
      case CompressionAlgos::kZLIB:
        return RCompressionSetting::EAlgorithm::kZLIB;
      case CompressionAlgos::kLZ4:
        return RCompressionSetting::EAlgorithm::kLZ4;
    }
    return RCompressionSetting::EAlgorithm::kZSTD;
  }
}  // namespace

using namespace ROOT::Experimental;
namespace edm {
  RNTupleOutputFile::RNTupleOutputFile(std::string const& iFileName,
                                       FileBlock const& iFileBlock,
                                       SelectedProductsForBranchType const& iSelected,
                                       Config const& iConfig,
                                       bool anyProductProduced)
      : file_(iFileName.c_str(), "recreate", ""),
        wrapperBaseTClass_(TClass::GetClass("edm::WrapperBase")),
        selectorConfig_(iConfig.selectorConfig),
        dropMetaData_(iConfig.dropMetaData) {
    setupRuns(iSelected[InRun], iConfig);
    setupLumis(iSelected[InLumi], iConfig);
    setupEvents(iSelected[InEvent], iConfig, anyProductProduced);
    setupPSets(iConfig);
    setupParentage(iConfig);
    setupMetaData(iConfig);

    auto const& branchToChildMap = iFileBlock.productDependencies().childLookup();
    for (auto const& parentToChildren : branchToChildMap) {
      for (auto const& child : parentToChildren.second) {
        productDependencies_.insertChild(parentToChildren.first, child);
      }
    }
  }

  namespace {
    std::string fixBranchName(std::string const& iName) {
      //need to remove the '.' at the end of the branch name
      return iName.substr(0, iName.size() - 1);
    }

    /* By default RNTuple will take a multi-byte intrinsic data type and break
it into multiple output fields to separate the high-bytes from the low-bytes (or mantessa from exponent).
This typically allows for better compression. Empirically we have found that some important
member data of some classes actually take more space on disk when this is done.
This function allows one to override the default RNTuple behavior and instead store
all bytes of a data type in one field. To do that one must find the storage type (typeName) and
explicitly pass the correct variable to `SetColumnRepresentatives`).
     */
    void noSplitField(ROOT::Experimental::RFieldBase& iField) {
      auto const& typeName = iField.GetTypeName();
      if (typeName == "std::uint16_t") {
        iField.SetColumnRepresentatives({{ROOT::Experimental::EColumnType::kUInt16}});
      } else if (typeName == "std::uint32_t") {
        iField.SetColumnRepresentatives({{ROOT::Experimental::EColumnType::kUInt32}});
      } else if (typeName == "std::uint64_t") {
        iField.SetColumnRepresentatives({{ROOT::Experimental::EColumnType::kUInt64}});
      } else if (typeName == "std::int16_t") {
        iField.SetColumnRepresentatives({{ROOT::Experimental::EColumnType::kInt16}});
      } else if (typeName == "std::int32_t") {
        iField.SetColumnRepresentatives({{ROOT::Experimental::EColumnType::kInt32}});
      } else if (typeName == "std::int64_t") {
        iField.SetColumnRepresentatives({{ROOT::Experimental::EColumnType::kInt64}});
      } else if (typeName == "float") {
        iField.SetColumnRepresentatives({{ROOT::Experimental::EColumnType::kReal32}});
      } else if (typeName == "double") {
        iField.SetColumnRepresentatives({{ROOT::Experimental::EColumnType::kReal64}});
      }
    }

    void findSubFieldsForNoSplitThenApply(ROOT::Experimental::RFieldBase& iField,
                                          std::vector<std::string> const& iNoSplitFields) {
      for (auto const& name : iNoSplitFields) {
        if (name.starts_with(iField.GetFieldName())) {
          bool found = false;
          for (auto& subfield : iField) {
            if (subfield.GetQualifiedFieldName() == name) {
              found = true;
              noSplitField(subfield);
              break;
            }
          }
          if (not found) {
            throw edm::Exception(edm::errors::Configuration)
                << "The data product was found but the requested subfield '" << name << "' is not part of the class";
          }
        }
      }
    }
  }  // namespace

  void RNTupleOutputFile::setupDataProducts(SelectedProducts const& iProducts,
                                            std::vector<bool> const& iUseStreamer,
                                            std::vector<std::string> const& iNoSplitFields,
                                            RNTupleModel& iModel) {
    unsigned int index = 0;
    const bool noSplitSubFields = (iNoSplitFields.size() == 1 and iNoSplitFields[0] == "all") ? true : false;
    for (auto const& prod : iProducts) {
      try {
        edm::convertException::wrap([&]() {
          if (index >= iUseStreamer.size() or not iUseStreamer[index]) {
            auto field = ROOT::Experimental::RFieldBase::Create(fixBranchName(prod.first->branchName()),
                                                                prod.first->wrappedName())
                             .Unwrap();
            if (noSplitSubFields) {
              //use the 'conventional' way to store fields
              for (auto& subfield : *field) {
                noSplitField(subfield);
              }
            } else if (not iNoSplitFields.empty()) {
              findSubFieldsForNoSplitThenApply(*field, iNoSplitFields);
            }
            iModel.AddField(std::move(field));
          } else {
            auto field = std::make_unique<ROOT::Experimental::RStreamerField>(fixBranchName(prod.first->branchName()),
                                                                              prod.first->wrappedName());
            iModel.AddField(std::move(field));
          }
          branchesWithStoredHistory_.insert(prod.first->branchID());
        });
        ++index;
      } catch (cms::Exception& iExcept) {
        using namespace std::string_literals;
        iExcept.addContext("while setting up field "s + prod.first->branchName());
        throw;
      }
    }
  }

  std::vector<RNTupleOutputFile::Product> RNTupleOutputFile::associateDataProducts(SelectedProducts const& iProducts,
                                                                                   RNTupleModel const& iModel) {
    std::vector<Product> ret;
    ret.reserve(iProducts.size());
    for (auto const& prod : iProducts) {
      ret.emplace_back(prod.second, prod.first, iModel.GetToken(fixBranchName(prod.first->branchName())));
    }
    return ret;
  }

  std::unique_ptr<RNTupleModel> RNTupleOutputFile::setupCommonModels(SelectedProducts const& iProducts,
                                                                     std::string const& iAuxName,
                                                                     std::string const& iAuxType) {
    auto model = ROOT::Experimental::RNTupleModel::CreateBare();
    {
      auto field = ROOT::Experimental::RFieldBase::Create(iAuxName, iAuxType).Unwrap();
      model->AddField(std::move(field));
    }
    const std::vector<bool> streamerNothing;
    const std::vector<std::string> unsplitNothing;
    setupDataProducts(iProducts, streamerNothing, unsplitNothing, *model);
    return model;
  }

  void RNTupleOutputFile::setupRuns(SelectedProducts const& iProducts, Config const& iConfig) {
    std::string kRunAuxName = "RunAuxiliary";
    {
      auto model = setupCommonModels(iProducts, "RunAuxiliary", "edm::RunAuxiliary");

      auto writeOptions = ROOT::Experimental::RNTupleWriteOptions();
      writeOptions.SetCompression(convert(iConfig.compressionAlgo), iConfig.compressionLevel);
      runs_ = ROOT::Experimental::RNTupleWriter::Append(std::move(model), "Runs", file_, writeOptions);
    }
    products_[InRun] = associateDataProducts(iProducts, runs_->GetModel());
    runAuxField_ = runs_->GetModel().GetToken(kRunAuxName);
  }
  void RNTupleOutputFile::setupLumis(SelectedProducts const& iProducts, Config const& iConfig) {
    std::string kLumiAuxName = "LuminosityBlockAuxiliary";
    {
      auto model = setupCommonModels(iProducts, "LuminosityBlockAuxiliary", "edm::LuminosityBlockAuxiliary");

      auto writeOptions = ROOT::Experimental::RNTupleWriteOptions();
      writeOptions.SetCompression(convert(iConfig.compressionAlgo), iConfig.compressionLevel);
      lumis_ = ROOT::Experimental::RNTupleWriter::Append(std::move(model), "LuminosityBlocks", file_, writeOptions);
    }
    products_[InLumi] = associateDataProducts(iProducts, lumis_->GetModel());
    lumiAuxField_ = lumis_->GetModel().GetToken(kLumiAuxName);
  }
  void RNTupleOutputFile::setupEvents(SelectedProducts const& iProducts,
                                      Config const& iConfig,
                                      bool anyProductProduced) {
    std::string kEventAuxName = "EventAuxiliary";
    std::string kEventProvName = "EventProductProvenance";
    std::string kEventSelName = "EventSelections";
    std::string kBranchListName = "BranchListIndexes";
    {
      auto model = ROOT::Experimental::RNTupleModel::CreateBare();
      {
        auto field = ROOT::Experimental::RFieldBase::Create(kEventAuxName, "edm::EventAuxiliary").Unwrap();
        model->AddField(std::move(field));
      }
      {
        auto field =
            ROOT::Experimental::RFieldBase::Create(kEventProvName, "std::vector<edm::StoredProductProvenance>").Unwrap();
        model->AddField(std::move(field));
      }
      {
        auto field = ROOT::Experimental::RFieldBase::Create(kEventSelName, "std::vector<edm::Hash<1> >").Unwrap();
        model->AddField(std::move(field));
      }
      {
        auto field = ROOT::Experimental::RFieldBase::Create(kBranchListName, "std::vector<unsigned short>").Unwrap();
        model->AddField(std::move(field));
      }
      setupDataProducts(iProducts, iConfig.streamerProduct, iConfig.doNotSplitSubFields, *model);

      auto writeOptions = ROOT::Experimental::RNTupleWriteOptions();
      writeOptions.SetCompression(convert(iConfig.compressionAlgo), iConfig.compressionLevel);
      writeOptions.SetApproxZippedClusterSize(iConfig.approxZippedClusterSize);
      writeOptions.SetMaxUnzippedClusterSize(iConfig.maxUnzippedClusterSize);
      writeOptions.SetInitialUnzippedPageSize(iConfig.initialUnzippedPageSize);
      writeOptions.SetMaxUnzippedPageSize(iConfig.maxUnzippedPageSize);
      writeOptions.SetPageBufferBudget(iConfig.pageBufferBudget);
      writeOptions.SetUseBufferedWrite(iConfig.useBufferedWrite);
      writeOptions.SetUseDirectIO(iConfig.useDirectIO);
      events_ = ROOT::Experimental::RNTupleWriter::Append(std::move(model), "Events", file_, writeOptions);
    }
    products_[InEvent] = associateDataProducts(iProducts, events_->GetModel());

    eventAuxField_ = events_->GetModel().GetToken(kEventAuxName);
    eventProvField_ = events_->GetModel().GetToken(kEventProvName);
    eventSelField_ = events_->GetModel().GetToken(kEventSelName);
    branchListField_ = events_->GetModel().GetToken(kBranchListName);

    // Note: The EventSelectionIDVector should have a one to one correspondence with the processes in the process history.
    // Therefore, a new entry should be added if and only if the current process has been added to the process history,
    // which is done if and only if there is a produced product.
    extendSelectorConfig_ = anyProductProduced || !iConfig.wantAllEvents;
  }
  void RNTupleOutputFile::setupPSets(Config const& iConfig) {
    auto model = ROOT::Experimental::RNTupleModel::CreateBare();
    {
      auto field = ROOT::Experimental::RFieldBase::Create("IdToParameterSetsBlobs",
                                                          "std::pair<edm::Hash<1>,edm::ParameterSetBlob>")
                       .Unwrap();
      model->AddField(std::move(field));
    }
    auto writeOptions = ROOT::Experimental::RNTupleWriteOptions();
    writeOptions.SetCompression(convert(iConfig.compressionAlgo), iConfig.compressionLevel);
    parameterSets_ = ROOT::Experimental::RNTupleWriter::Append(std::move(model), "ParameterSets", file_, writeOptions);
  }

  void RNTupleOutputFile::fillPSets() {
    std::pair<ParameterSetID, ParameterSetBlob> idToBlob;

    auto rentry = parameterSets_->CreateEntry();
    rentry->BindRawPtr("IdToParameterSetsBlobs", static_cast<void*>(&idToBlob));

    for (auto const& pset : *pset::Registry::instance()) {
      idToBlob.first = pset.first;
      idToBlob.second.pset() = pset.second.toString();

      parameterSets_->Fill(*rentry);
    }
  }

  void RNTupleOutputFile::setupParentage(Config const& iConfig) {
    auto model = ROOT::Experimental::RNTupleModel::CreateBare();
    {
      auto field = ROOT::Experimental::RFieldBase::Create("Description", "edm::Parentage").Unwrap();
      model->AddField(std::move(field));
    }
    auto writeOptions = ROOT::Experimental::RNTupleWriteOptions();
    writeOptions.SetCompression(convert(iConfig.compressionAlgo), iConfig.compressionLevel);
    parentage_ = ROOT::Experimental::RNTupleWriter::Append(std::move(model), "Parentage", file_, writeOptions);
  }
  void RNTupleOutputFile::fillParentage() {
    ParentageRegistry& ptReg = *ParentageRegistry::instance();

    std::vector<ParentageID> orderedIDs(parentageIDs_.size());
    for (auto const& parentageID : parentageIDs_) {
      orderedIDs[parentageID.second] = parentageID.first;
    }

    auto rentry = parentage_->CreateEntry();
    //now put them into the RNTuple in the correct order
    for (auto const& orderedID : orderedIDs) {
      auto desc = ptReg.getMapped(orderedID);
      rentry->BindRawPtr("Description", const_cast<void*>(static_cast<void const*>(desc)));
      parentage_->Fill(*rentry);
    }
  }

  void RNTupleOutputFile::setupMetaData(Config const& iConfig) {
    auto model = ROOT::Experimental::RNTupleModel::CreateBare();
    {
      //Ultimately will need a new class specific for RNTuple
      //auto field = ROOT::Experimental::RFieldBase::Create("FileFormatVersion", "edm::FileFormatVersion").Unwrap();
      //model->AddField(std::move(field));
    }
    {
      auto field = ROOT::Experimental::RFieldBase::Create("FileIdentifier", "edm::FileID").Unwrap();
      model->AddField(std::move(field));
    }

    {
      auto field = ROOT::Experimental::RFieldBase::Create("IndexIntoFile", "edm::IndexIntoFile").Unwrap();
      model->AddField(std::move(field));
    }
    {
      auto field = ROOT::Experimental::RFieldBase::Create("MergeableRunProductMetadata",
                                                          "edm::StoredMergeableRunProductMetadata")
                       .Unwrap();
      model->AddField(std::move(field));
    }
    {
      auto field =
          ROOT::Experimental::RFieldBase::Create("ProcessHistory", "std::vector<edm::ProcessHistory>").Unwrap();
      model->AddField(std::move(field));
    }
    {
      auto field = ROOT::Experimental::RFieldBase::Create("ProductRegistry", "edm::ProductRegistry").Unwrap();
      model->AddField(std::move(field));
    }
    {
      auto field =
          ROOT::Experimental::RFieldBase::Create("BranchIDLists", "std::vector<std::vector<unsigned int> >").Unwrap();
      model->AddField(std::move(field));
    }
    {
      auto field =
          ROOT::Experimental::RFieldBase::Create("ThinnedAssociationsHelper", "edm::ThinnedAssociationsHelper").Unwrap();
      model->AddField(std::move(field));
    }
    {
      auto field = ROOT::Experimental::RFieldBase::Create("ProductDependencies", "edm::BranchChildren").Unwrap();
      model->AddField(std::move(field));
    }

    auto writeOptions = ROOT::Experimental::RNTupleWriteOptions();
    writeOptions.SetCompression(convert(iConfig.compressionAlgo), iConfig.compressionLevel);
    metaData_ = ROOT::Experimental::RNTupleWriter::Append(std::move(model), "MetaData", file_, writeOptions);
  }

  void RNTupleOutputFile::fillMetaData(BranchIDLists const& iBranchIDLists,
                                       ThinnedAssociationsHelper const& iThinnedHelper,
                                       ProductRegistry const& iReg) {
    auto rentry = metaData_->CreateEntry();

    FileID id(createGlobalIdentifier());

    rentry->BindRawPtr("FileIdentifier", &id);

    indexIntoFile_.sortVector_Run_Or_Lumi_Entries();
    rentry->BindRawPtr("IndexIntoFile", &indexIntoFile_);

    ProcessHistoryVector procHistoryVector;
    for (auto const& ph : processHistoryRegistry_) {
      procHistoryVector.push_back(ph.second);
    }
    rentry->BindRawPtr("ProcessHistory", &procHistoryVector);

    // Make a local copy of the ProductRegistry, removing any transient or pruned products.
    using ProductList = ProductRegistry::ProductList;
    ProductRegistry pReg(iReg.productList());
    ProductList& pList = const_cast<ProductList&>(pReg.productList());
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

    rentry->BindRawPtr("ProductRegistry", &pReg);
    rentry->BindRawPtr("ThinnedAssociationsHelper", const_cast<void*>(static_cast<const void*>(&iThinnedHelper)));
    rentry->BindRawPtr("BranchIDLists", const_cast<void*>(static_cast<const void*>(&iBranchIDLists)));
    rentry->BindRawPtr("ProductDependencies", const_cast<void*>(static_cast<const void*>(&productDependencies_)));

    metaData_->Fill(*rentry);
  }

  void RNTupleOutputFile::openFile(FileBlock const& fb) {
    auto const& branchToChildMap = fb.productDependencies().childLookup();
    for (auto const& parentToChildren : branchToChildMap) {
      for (auto const& child : parentToChildren.second) {
        productDependencies_.insertChild(parentToChildren.first, child);
      }
    }
  }

  void RNTupleOutputFile::reallyCloseFile(BranchIDLists const& iBranchIDLists,
                                          ThinnedAssociationsHelper const& iThinnedHelper,
                                          ProductRegistry const& iReg) {
    fillPSets();
    fillParentage();
    fillMetaData(iBranchIDLists, iThinnedHelper, iReg);
  }

  RNTupleOutputFile::~RNTupleOutputFile() {}

  std::vector<std::unique_ptr<WrapperBase>> RNTupleOutputFile::writeDataProducts(std::vector<Product> const& iProducts,
                                                                                 OccurrenceForOutput const& iOccurence,
                                                                                 REntry& iEntry) {
    std::vector<std::unique_ptr<WrapperBase>> dummies;

    for (auto const& p : iProducts) {
      auto h = iOccurence.getByToken(p.get_, p.desc_->unwrappedTypeID());
      auto product = h.wrapper();
      if (nullptr == product) {
        // No product with this ID is in the event.
        // Add a null product.
        TClass* cp = p.desc_->wrappedType().getClass();
        assert(cp != nullptr);
        int offset = cp->GetBaseClassOffset(wrapperBaseTClass_);
        void* p = cp->New();
        std::unique_ptr<WrapperBase> dummy = getWrapperBasePtr(p, offset);
        product = dummy.get();
        dummies.emplace_back(std::move(dummy));
      }
      iEntry.BindRawPtr(p.field_, const_cast<void*>(static_cast<void const*>(product)));
    }
    return dummies;
  }

  std::vector<StoredProductProvenance> RNTupleOutputFile::writeDataProductProvenance(
      std::vector<Product> const& iProducts, EventForOutput const& iEvent) {
    std::set<StoredProductProvenance> provenanceToKeep;

    if (not dropMetaData_) {
      for (auto const& p : iProducts) {
        auto h = iEvent.getByToken(p.get_, p.desc_->unwrappedTypeID());
        if (h.isValid()) {
          auto prov = h.provenance()->productProvenance();
          if (not prov) {
            prov = iEvent.productProvenanceRetrieverPtr()->branchIDToProvenance(p.desc_->originalBranchID());
          }
          if (prov) {
            insertProductProvenance(*prov, provenanceToKeep);
          }
        }
      }
    }
    return std::vector<StoredProductProvenance>(provenanceToKeep.begin(), provenanceToKeep.end());
  }

  bool RNTupleOutputFile::insertProductProvenance(ProductProvenance const& iProv,
                                                  std::set<StoredProductProvenance>& oToInsert) {
    StoredProductProvenance toStore;
    toStore.branchID_ = iProv.branchID().id();
    auto itFound = oToInsert.find(toStore);
    if (itFound == oToInsert.end()) {
      //get the index to the ParentageID or insert a new value if not already present
      auto i = parentageIDs_.emplace(iProv.parentageID(), static_cast<unsigned int>(parentageIDs_.size()));
      toStore.parentageIDIndex_ = i.first->second;
      if (toStore.parentageIDIndex_ >= parentageIDs_.size()) {
        throw edm::Exception(errors::LogicError)
            << "RNTupleOutputFile::insertProductProvenance\n"
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

  void RNTupleOutputFile::insertAncestorsProvenance(ProductProvenance const& iProv,
                                                    ProductProvenanceRetriever const& iMapper,
                                                    std::set<StoredProductProvenance>& oToKeep) {
    std::vector<BranchID> const& parentIDs = iProv.parentage().parents();
    for (auto const& parentID : parentIDs) {
      branchesWithStoredHistory_.insert(parentID);
      ProductProvenance const* info = iMapper.branchIDToProvenance(parentID);
      if (info) {
        if (insertProductProvenance(*info, oToKeep)) {
          //haven't seen this one yet
          insertAncestorsProvenance(*info, iMapper, oToKeep);
        }
      }
    }
  }

  void RNTupleOutputFile::write(EventForOutput const& e) {
    {
      auto rentry = events_->CreateEntry();
      rentry->BindRawPtr(*eventAuxField_, const_cast<void*>(static_cast<void const*>(&(e.eventAuxiliary()))));
      rentry->BindRawPtr(*eventSelField_, const_cast<void*>(static_cast<void const*>(&(e.eventSelectionIDs()))));
      rentry->BindRawPtr(*branchListField_, const_cast<void*>(static_cast<void const*>(&(e.branchListIndexes()))));

      EventSelectionIDVector esids = e.eventSelectionIDs();
      if (extendSelectorConfig_) {
        esids.push_back(selectorConfig_);
      }
      rentry->BindRawPtr(*eventSelField_, &esids);

      auto dummies = writeDataProducts(products_[InEvent], e, *rentry);
      auto prov = writeDataProductProvenance(products_[InEvent], e);
      rentry->BindRawPtr(*eventProvField_, &prov);
      events_->Fill(*rentry);
    }

    processHistoryRegistry_.registerProcessHistory(e.processHistory());
    // Store the reduced ID in the IndexIntoFile
    ProcessHistoryID reducedPHID = processHistoryRegistry_.reducedProcessHistoryID(e.processHistoryID());
    // Add event to index
    indexIntoFile_.addEntry(reducedPHID, e.run(), e.luminosityBlock(), e.event(), eventEntryNumber_);
    ++eventEntryNumber_;
  }

  void RNTupleOutputFile::writeLuminosityBlock(LuminosityBlockForOutput const& iLumi) {
    {
      auto rentry = lumis_->CreateEntry();
      auto lumiAux = iLumi.luminosityBlockAuxiliary();
      lumiAux.setProcessHistoryID(iLumi.processHistoryID());
      rentry->BindRawPtr(*lumiAuxField_, static_cast<void*>(&lumiAux));
      auto dummies = writeDataProducts(products_[InLumi], iLumi, *rentry);
      lumis_->Fill(*rentry);
    }
    processHistoryRegistry_.registerProcessHistory(iLumi.processHistory());
    // Store the reduced ID in the IndexIntoFile
    ProcessHistoryID reducedPHID = processHistoryRegistry_.reducedProcessHistoryID(iLumi.processHistoryID());
    // Add lumi to index.
    indexIntoFile_.addEntry(
        reducedPHID, iLumi.run(), iLumi.luminosityBlock(), IndexIntoFile::invalidEvent, lumiEntryNumber_);
    ++lumiEntryNumber_;
  }

  void RNTupleOutputFile::writeRun(RunForOutput const& iRun) {
    {
      auto rentry = runs_->CreateEntry();
      auto runAux = iRun.runAuxiliary();
      runAux.setProcessHistoryID(iRun.processHistoryID());
      rentry->BindRawPtr(*runAuxField_, static_cast<void*>(&runAux));
      auto dummies = writeDataProducts(products_[InRun], iRun, *rentry);
      runs_->Fill(*rentry);
    }
    processHistoryRegistry_.registerProcessHistory(iRun.processHistory());
    // Store the reduced ID in the IndexIntoFile
    ProcessHistoryID reducedPHID = processHistoryRegistry_.reducedProcessHistoryID(iRun.processHistoryID());
    // Add run to index.
    indexIntoFile_.addEntry(
        reducedPHID, iRun.run(), IndexIntoFile::invalidLumi, IndexIntoFile::invalidEvent, runEntryNumber_);
    ++runEntryNumber_;
  }

}  // namespace edm
