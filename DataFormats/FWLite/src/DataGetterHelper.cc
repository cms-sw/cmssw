// -*- C++ -*-
//
// Package:     DataFormats/FWLite
// Class  :     DataGetterHelper
//
// Implementation:
//     [Notes on implementation]
//
// Original Author: Eric Vaandering
//         Created:  Fri Jan 29 11:58:01 CST 2010
//

// system include files
#include <cassert>
#include <iostream>

// user include files
#include "DataFormats/FWLite/interface/DataGetterHelper.h"
#include "TFile.h"
#include "TTree.h"
#include "TTreeCache.h"

#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/Common/interface/getThinned_implementation.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"

#include "FWCore/FWLite/interface/setRefStreamer.h"

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Reflection/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

namespace fwlite {
  //
  // constants, enums and typedefs
  //

  //
  // static data member definitions
  //
  // empty object used to signal that the branch requested was not found
  static internal::Data branchNotFound{};
  static const char kEmptyString[1] = {0};

  //
  // constructors and destructor
  //
  DataGetterHelper::DataGetterHelper(TTree* tree,
                                     std::shared_ptr<HistoryGetterBase> historyGetter,
                                     std::shared_ptr<BranchMapReader> branchMap,
                                     std::shared_ptr<edm::EDProductGetter> getter,
                                     bool useCache,
                                     std::function<void(TBranch const&)> baFunc)
      : branchMap_(branchMap),
        historyGetter_(historyGetter),
        getter_(getter),
        tcTrained_(false),
        tcUse_(useCache),
        branchAccessFunc_(baFunc) {
    if (nullptr == tree) {
      throw cms::Exception("NoTree") << "The TTree pointer passed to the constructor was null";
    }
    tree_ = tree;
    if (tcUse_) {
      tree_->SetCacheSize();
    }
  }

  // DataGetterHelper::DataGetterHelper(const DataGetterHelper& rhs)
  // {
  //    // do actual copying here;
  // }

  DataGetterHelper::~DataGetterHelper() {}

  //
  // assignment operators
  //
  // const DataGetterHelper& DataGetterHelper::operator=(const DataGetterHelper& rhs)
  // {
  //   //An exception safe implementation is
  //   DataGetterHelper temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }

  //
  // member functions
  //

  //
  // const member functions
  //

  //
  // static member functions
  //

  static TBranch* findBranch(TTree* iTree, std::string const& iMainLabels, std::string const& iProcess) {
    std::string branchName(iMainLabels);
    branchName += iProcess;
    //branchName+=".obj";
    branchName += ".";
    return iTree->GetBranch(branchName.c_str());
  }

  void DataGetterHelper::getBranchData(edm::EDProductGetter const* iGetter,
                                       Long64_t eventEntry,
                                       internal::Data& iData) const {
    GetterOperate op(iGetter);

    ////WORK AROUND FOR ROOT!!
    ////Create a new instance so that we can clear any cache the object uses
    ////this slows the code down
    //edm::ObjectWithDict obj = iData.obj_;
    //iData.obj_ = iData.obj_.construct();
    //iData.pObj_ = iData.obj_.address();
    //iData.branch_->SetAddress(&(iData.pObj_));
    ////If a REF to this was requested in the past, we might as well do the work now
    //if(nullptr != iData.pProd_) {
    //    iData.pProd_ = iData.obj_.address();
    //}
    //obj.destruct();
    ////END OF WORK AROUND

    if (tcUse_) {
      TTreeCache* tcache = dynamic_cast<TTreeCache*>(branchMap_->getFile()->GetCacheRead());

      if (nullptr != tcache) {
        if (!tcTrained_) {
          tcache->SetLearnEntries(100);
          tcache->SetEntryRange(0, tree_->GetEntries());
          tcTrained_ = true;
        }
        tree_->LoadTree(eventEntry);
      }
    }
    branchAccessFunc_(*iData.branch_);
    iData.branch_->GetEntry(eventEntry);

    iData.lastProduct_ = eventEntry;
  }

  std::optional<edm::BranchID> DataGetterHelper::getBranchIDFor(std::type_info const& iInfo,
                                                                char const* iModuleLabel,
                                                                char const* iProductInstanceLabel,
                                                                char const* iProcessLabel) const {
    auto branchFor = [this](edm::TypeID const& iInfo,
                            char const* iModuleLabel,
                            char const* iProductInstanceLabel,
                            char const* iProcessLabel) -> std::optional<edm::BranchID> {
      for (auto const& bd : branchMap_->getBranchDescriptions()) {
        if (bd.unwrappedTypeID() == iInfo and bd.moduleLabel() == iModuleLabel and
            bd.productInstanceName() == iProductInstanceLabel and bd.processName() == iProcessLabel) {
          return bd.branchID();
        }
      }
      return std::nullopt;
    };
    if (nullptr == iProcessLabel || strlen(iProcessLabel) == 0) {
      //have to search in reverse order since newest are on the bottom
      const edm::ProcessHistory& h = DataGetterHelper::history();
      edm::TypeID typeID(iInfo);
      for (edm::ProcessHistory::const_reverse_iterator iproc = h.rbegin(), eproc = h.rend(); iproc != eproc; ++iproc) {
        auto v = branchFor(typeID, iModuleLabel, iProductInstanceLabel, iproc->processName().c_str());
        if (v) {
          return v;
        }
      }
      return std::nullopt;
    }
    return branchFor(edm::TypeID(iInfo), iModuleLabel, iProductInstanceLabel, iProcessLabel);
  }

  internal::Data& DataGetterHelper::getBranchDataFor(std::type_info const& iInfo,
                                                     char const* iModuleLabel,
                                                     char const* iProductInstanceLabel,
                                                     char const* iProcessLabel) const {
    edm::TypeID type(iInfo);
    internal::DataKey key(type, iModuleLabel, iProductInstanceLabel, iProcessLabel);

    KeyToDataMap::iterator itFind = data_.find(key);
    if (itFind == data_.end()) {
      //see if such a branch actually exists
      std::string const sep("_");
      //CHANGE: If this fails, need to lookup the the friendly name which was used to write the file
      std::string name(type.friendlyClassName());
      name += sep + std::string(key.module());
      name += sep + std::string(key.product()) + sep;

      //if we have to lookup the process label, remember it and register the product again
      std::string foundProcessLabel;
      TBranch* branch = nullptr;
      std::shared_ptr<internal::Data> theData;

      if (nullptr == iProcessLabel || iProcessLabel == key.kEmpty() || strlen(iProcessLabel) == 0) {
        std::string const* lastLabel = nullptr;
        //have to search in reverse order since newest are on the bottom
        const edm::ProcessHistory& h = DataGetterHelper::history();
        for (edm::ProcessHistory::const_reverse_iterator iproc = h.rbegin(), eproc = h.rend(); iproc != eproc;
             ++iproc) {
          lastLabel = &(iproc->processName());
          branch = findBranch(tree_, name, iproc->processName());
          if (nullptr != branch) {
            break;
          }
        }
        if (nullptr == branch) {
          return branchNotFound;
        }
        //do we already have this one?
        if (nullptr != lastLabel) {
          internal::DataKey fullKey(type, iModuleLabel, iProductInstanceLabel, lastLabel->c_str());
          itFind = data_.find(fullKey);
          if (itFind != data_.end()) {
            //remember the data we've found
            theData = itFind->second;
          } else {
            //only set this if we don't already have it since it this string is not empty we re-register
            foundProcessLabel = *lastLabel;
          }
        }
      } else {
        //we have all the pieces
        branch = findBranch(tree_, name, key.process());
        if (nullptr == branch) {
          return branchNotFound;
        }
      }

      //cache the info
      size_t moduleLabelLen = strlen(iModuleLabel) + 1;
      char* newModule = new char[moduleLabelLen];
      std::strncpy(newModule, iModuleLabel, moduleLabelLen);
      labels_.push_back(newModule);

      char const* newProduct = kEmptyString;
      if (key.product()[0] != 0) {
        size_t newProductLen = strlen(key.product()) + 1;
        auto newProductTmp = new char[newProductLen];
        std::strncpy(newProductTmp, key.product(), newProductLen);
        newProduct = newProductTmp;
        labels_.push_back(newProduct);
      }
      char const* newProcess = kEmptyString;
      if (key.process()[0] != 0) {
        size_t newProcessLen = strlen(key.process()) + 1;
        auto newProcessTmp = new char[newProcessLen];
        std::strncpy(newProcessTmp, key.process(), newProcessLen);
        newProcess = newProcessTmp;
        labels_.push_back(newProcess);
      }
      internal::DataKey newKey(edm::TypeID(iInfo), newModule, newProduct, newProcess);

      if (nullptr == theData.get()) {
        //We do not already have this data as another key

        //create an instance of the object to be used as a buffer
        edm::TypeWithDict typeWithDict(iInfo);
        if (!bool(typeWithDict)) {
          throw cms::Exception("UnknownType") << "No dictionary exists for type " << iInfo.name();
        }

        edm::ObjectWithDict obj = edm::ObjectWithDict::byType(typeWithDict);

        if (obj.address() == nullptr) {
          throw cms::Exception("ConstructionFailed") << "failed to construct an instance of " << typeWithDict.name();
        }
        auto newData = std::make_shared<internal::Data>();
        newData->branch_ = branch;
        newData->obj_ = obj;
        newData->lastProduct_ = -1;
        newData->pObj_ = obj.address();
        newData->pProd_ = nullptr;
        branch->SetAddress(&(newData->pObj_));
        theData = newData;
      }
      itFind = data_.insert(std::make_pair(newKey, theData)).first;

      if (!foundProcessLabel.empty()) {
        //also remember it with the process label
        auto newProcessTmp = new char[foundProcessLabel.size() + 1];
        std::strcpy(newProcessTmp, foundProcessLabel.c_str());
        newProcess = newProcessTmp;
        labels_.push_back(newProcess);
        internal::DataKey newKeyWithProcess(edm::TypeID(iInfo), newModule, newProduct, newProcess);

        data_.insert(std::make_pair(newKeyWithProcess, theData));
      }
    }
    return *(itFind->second);
  }

  std::string const DataGetterHelper::getBranchNameFor(std::type_info const& iInfo,
                                                       char const* iModuleLabel,
                                                       char const* iProductInstanceLabel,
                                                       char const* iProcessLabel) const {
    internal::Data& theData =
        DataGetterHelper::getBranchDataFor(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel);

    if (nullptr != theData.branch_) {
      return std::string(theData.branch_->GetName());
    }
    return std::string("");
  }

  bool DataGetterHelper::getByLabel(std::type_info const& iInfo,
                                    char const* iModuleLabel,
                                    char const* iProductInstanceLabel,
                                    char const* iProcessLabel,
                                    void* oData,
                                    Long_t eventEntry) const {
    // Maintain atEnd() check in parent classes
    void** pOData = reinterpret_cast<void**>(oData);
    *pOData = nullptr;

    internal::Data& theData =
        DataGetterHelper::getBranchDataFor(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel);

    if (nullptr != theData.branch_) {
      if (eventEntry != theData.lastProduct_) {
        //haven't gotten the data for this event
        getBranchData(getter_.get(), eventEntry, theData);
      }
      *pOData = theData.obj_.address();
    }

    if (nullptr == *pOData)
      return false;
    else
      return true;
  }

  bool DataGetterHelper::getByBranchDescription(edm::BranchDescription const& bDesc,
                                                Long_t eventEntry,
                                                KeyToDataMap::iterator& itData) const {
    if (!bDesc.branchID().isValid()) {
      return false;
    }

    //Calculate the key from the branch description
    edm::TypeWithDict typeWD(edm::TypeWithDict::byName(edm::wrappedClassName(bDesc.fullClassName())));
    edm::TypeID type(typeWD.typeInfo());
    assert(bool(type));

    //Only the product instance label may be empty
    char const* pIL = bDesc.productInstanceName().c_str();
    if (pIL[0] == 0) {
      pIL = nullptr;
    }
    internal::DataKey k(type, bDesc.moduleLabel().c_str(), pIL, bDesc.processName().c_str());

    //has this already been gotten?
    itData = data_.find(k);
    if (data_.end() == itData) {
      //ask for the data
      edm::WrapperBase const* dummy = nullptr;
      getByLabel(type.typeInfo(), k.module(), k.product(), k.process(), &dummy, eventEntry);
      if (nullptr == dummy) {
        return false;
      }
      itData = data_.find(k);
      assert(itData != data_.end());
      assert(dummy == itData->second->obj_.address());
    }
    return true;
  }

  edm::WrapperBase const* DataGetterHelper::getByProductID(edm::ProductID const& iID, Long_t eventEntry) const {
    typedef std::pair<edm::ProductID, edm::BranchListIndex> IDPair;
    IDPair theID = std::make_pair(iID, branchMap_->branchListIndexes()[iID.processIndex() - 1]);
    std::map<IDPair, std::shared_ptr<internal::Data>>::const_iterator itFound = idToData_.find(theID);

    if (itFound == idToData_.end()) {
      edm::BranchDescription const& bDesc = branchMap_->productToBranch(iID);
      KeyToDataMap::iterator itData;

      if (!getByBranchDescription(bDesc, eventEntry, itData)) {
        return nullptr;
      }
      itFound = idToData_.insert(std::make_pair(theID, itData->second)).first;
    }
    if (eventEntry != itFound->second->lastProduct_) {
      //haven't gotten the data for this event
      getBranchData(getter_.get(), eventEntry, *(itFound->second));
    }
    if (nullptr == itFound->second->pProd_) {
      itFound->second->pProd_ = wrapperBasePtr(itFound->second->obj_);
      if (nullptr == itFound->second->pProd_) {
        return nullptr;
      }
    }
    return itFound->second->pProd_;
  }

  edm::WrapperBase const* DataGetterHelper::getByBranchID(edm::BranchID const& bid, Long_t eventEntry) const {
    auto itFound = bidToData_.find(bid);

    if (itFound == bidToData_.end()) {
      edm::BranchDescription const& bDesc = branchMap_->branchIDToBranch(bid);
      KeyToDataMap::iterator itData;

      if (!getByBranchDescription(bDesc, eventEntry, itData)) {
        return nullptr;
      }
      itFound = bidToData_.insert(std::make_pair(bid, itData->second)).first;
    }
    if (eventEntry != itFound->second->lastProduct_) {
      //haven't gotten the data for this event
      getBranchData(getter_.get(), eventEntry, *(itFound->second));
    }
    if (nullptr == itFound->second->pProd_) {
      itFound->second->pProd_ = wrapperBasePtr(itFound->second->obj_);
      if (nullptr == itFound->second->pProd_) {
        return nullptr;
      }
    }
    return itFound->second->pProd_;
  }

  edm::WrapperBase const* DataGetterHelper::wrapperBasePtr(edm::ObjectWithDict const& objectWithDict) const {
    // This converts a void* that points at a Wrapper<T>* into a WrapperBase*
    edm::TypeWithDict wrapperBaseTypeWithDict(typeid(edm::WrapperBase));
    return static_cast<edm::WrapperBase const*>(
        wrapperBaseTypeWithDict.pointerToBaseType(objectWithDict.address(), objectWithDict.typeOf()));
  }

  std::optional<std::tuple<edm::WrapperBase const*, unsigned int>> DataGetterHelper::getThinnedProduct(
      edm::ProductID const& pid, unsigned int key, Long_t eventEntry) const {
    return edm::detail::getThinnedProduct(
        pid,
        key,
        branchMap_->thinnedAssociationsHelper(),
        [this](edm::ProductID const& p) { return branchMap_->productToBranchID(p); },
        [this, eventEntry](edm::BranchID const& b) { return getThinnedAssociation(b, eventEntry); },
        [this, eventEntry](edm::ProductID const& p) { return getByProductID(p, eventEntry); });
  }

  void DataGetterHelper::getThinnedProducts(edm::ProductID const& pid,
                                            std::vector<edm::WrapperBase const*>& foundContainers,
                                            std::vector<unsigned int>& keys,
                                            Long_t eventEntry) const {
    edm::detail::getThinnedProducts(
        pid,
        branchMap_->thinnedAssociationsHelper(),
        [this](edm::ProductID const& p) { return branchMap_->productToBranchID(p); },
        [this, eventEntry](edm::BranchID const& b) { return getThinnedAssociation(b, eventEntry); },
        [this, eventEntry](edm::ProductID const& p) { return getByProductID(p, eventEntry); },
        foundContainers,
        keys);
  }

  edm::OptionalThinnedKey DataGetterHelper::getThinnedKeyFrom(edm::ProductID const& parentID,
                                                              unsigned int key,
                                                              edm::ProductID const& thinnedID,
                                                              Long_t eventEntry) const {
    edm::BranchID parent = branchMap_->productToBranchID(parentID);
    if (!parent.isValid())
      return std::monostate{};
    edm::BranchID thinned = branchMap_->productToBranchID(thinnedID);
    if (!thinned.isValid())
      return std::monostate{};

    try {
      auto ret = edm::detail::getThinnedKeyFrom_implementation(
          parentID,
          parent,
          key,
          thinnedID,
          thinned,
          branchMap_->thinnedAssociationsHelper(),
          [this, eventEntry](edm::BranchID const& branchID) { return getThinnedAssociation(branchID, eventEntry); });
      if (auto factory = std::get_if<edm::detail::GetThinnedKeyFromExceptionFactory>(&ret)) {
        return [func = *factory]() {
          auto ex = func();
          ex.addContext("Calling DataGetterHelper::getThinnedKeyFrom()");
          return ex;
        };
      } else {
        return ret;
      }
    } catch (edm::Exception& ex) {
      ex.addContext("Calling DataGetterHelper::getThinnedKeyFrom()");
      throw ex;
    }
  }

  edm::ThinnedAssociation const* DataGetterHelper::getThinnedAssociation(edm::BranchID const& branchID,
                                                                         Long_t eventEntry) const {
    edm::WrapperBase const* wrapperBase = getByBranchID(branchID, eventEntry);
    if (wrapperBase == nullptr) {
      throw edm::Exception(edm::errors::LogicError)
          << "DataGetterHelper::getThinnedAssociation, product ThinnedAssociation not found.\n";
    }
    if (!(typeid(edm::ThinnedAssociation) == wrapperBase->dynamicTypeInfo())) {
      throw edm::Exception(edm::errors::LogicError)
          << "DataGetterHelper::getThinnedAssociation, product has wrong type, not a ThinnedAssociation.\n";
    }
    edm::Wrapper<edm::ThinnedAssociation> const* wrapper =
        static_cast<edm::Wrapper<edm::ThinnedAssociation> const*>(wrapperBase);

    edm::ThinnedAssociation const* thinnedAssociation = wrapper->product();
    return thinnedAssociation;
  }

  const edm::ProcessHistory& DataGetterHelper::history() const { return historyGetter_->history(); }

  //
  // static member functions
  //
}  // namespace fwlite
