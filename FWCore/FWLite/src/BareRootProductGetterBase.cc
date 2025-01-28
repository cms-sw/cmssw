// -*- C++ -*-
//
// Package:     FWLite
// Class  :     BareRootProductGetterBase
//
// Implementation:
//     <Notes on implementation>
//
// Description: This file was originally BareRootProductGetter.cc.
// It was copied to BareRootProductGetterBase.cc in order to refactor
// it a little bit to make it usable for FireworksWeb.
//
// Original Author:  Chris Jones
//         Created:  Tue May 23 11:03:31 EDT 2006
//

// user include files
#include "FWCore/FWLite/interface/BareRootProductGetterBase.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/getThinned_implementation.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

// system include files

#include "TBranch.h"
#include "TClass.h"
#include "TFile.h"
#include "TTree.h"

//
// constructors and destructor
//
BareRootProductGetterBase::BareRootProductGetterBase() = default;
BareRootProductGetterBase::~BareRootProductGetterBase() = default;

//
// const member functions
//
edm::WrapperBase const* BareRootProductGetterBase::getIt(edm::ProductID const& pid) const {
  // std::cout << "getIt called " << pid << std::endl;
  if (branchMap_.updateFile(currentFile())) {
    idToBuffers_.clear();
  }
  TTree* eventTree = branchMap_.getEventTree();
  // std::cout << "eventTree " << eventTree << std::endl;
  if (nullptr == eventTree) {
    throw cms::Exception("NoEventsTree")
        << "unable to find the TTree '" << edm::poolNames::eventTreeName() << "' in the last open file, \n"
        << "file: '" << branchMap_.getFile()->GetName()
        << "'\n Please check that the file is a standard CMS ROOT format.\n"
        << "If the above is not the file you expect then please open your data file after all other files.";
  }
  Long_t eventEntry = eventTree->GetReadEntry();
  // std::cout << "eventEntry " << eventEntry << std::endl;
  branchMap_.updateEvent(eventEntry);
  if (eventEntry < 0) {
    throw cms::Exception("GetEntryNotCalled")
        << "please call GetEntry for the 'Events' TTree for each event in order to make edm::Ref's work."
        << "\n Also be sure to call 'SetAddress' for all Branches after calling the GetEntry.";
  }

  edm::BranchID branchID = branchMap_.productToBranchID(pid);

  return getIt(branchID, eventEntry);
}

edm::WrapperBase const* BareRootProductGetterBase::getIt(edm::BranchID const& branchID, Long_t eventEntry) const {
  Buffer* buffer = nullptr;
  IdToBuffers::iterator itBuffer = idToBuffers_.find(branchID);

  // std::cout << "Buffers" << std::endl;
  if (itBuffer == idToBuffers_.end()) {
    buffer = createNewBuffer(branchID);
    // std::cout << "buffer " << buffer << std::endl;
    if (nullptr == buffer) {
      return nullptr;
    }
  } else {
    buffer = &(itBuffer->second);
  }
  if (nullptr == buffer) {
    throw cms::Exception("NullBuffer") << "Found a null buffer which is supposed to hold the data item."
                                       << "\n Please contact developers since this message should not happen.";
  }
  if (nullptr == buffer->branch_) {
    throw cms::Exception("NullBranch") << "The TBranch which should hold the data item is null."
                                       << "\n Please contact the developers since this message should not happen.";
  }
  if (buffer->eventEntry_ != eventEntry) {
    //NOTE: Need to reset address because user could have set the address themselves
    //std::cout << "new event" << std::endl;

    //ROOT WORKAROUND: Create new objects so any internal data cache will get cleared
    void* address = buffer->class_->New();

    static TClass const* edproductTClass = TClass::GetClass(typeid(edm::WrapperBase));
    edm::WrapperBase const* prod =
        static_cast<edm::WrapperBase const*>(buffer->class_->DynamicCast(edproductTClass, address, true));

    if (nullptr == prod) {
      cms::Exception("FailedConversion") << "failed to convert a '" << buffer->class_->GetName()
                                         << "' to a edm::WrapperBase."
                                         << "Please contact developers since something is very wrong.";
    }
    buffer->address_ = address;
    buffer->product_ = std::shared_ptr<edm::WrapperBase const>(prod);
    //END WORKAROUND

    address = &(buffer->address_);
    buffer->branch_->SetAddress(address);

    buffer->branch_->GetEntry(eventEntry);
    buffer->eventEntry_ = eventEntry;
  }
  if (!buffer->product_) {
    throw cms::Exception("BranchGetEntryFailed")
        << "Calling GetEntry with index " << eventEntry << "for branch " << buffer->branch_->GetName() << " failed.";
  }

  return buffer->product_.get();
}

std::optional<std::tuple<edm::WrapperBase const*, unsigned int>> BareRootProductGetterBase::getThinnedProduct(
    edm::ProductID const& pid, unsigned int key) const {
  Long_t eventEntry = branchMap_.getEventTree()->GetReadEntry();
  return edm::detail::getThinnedProduct(
      pid,
      key,
      branchMap_.thinnedAssociationsHelper(),
      [this](edm::ProductID const& p) { return branchMap_.productToBranchID(p); },
      [this, eventEntry](edm::BranchID const& b) { return getThinnedAssociation(b, eventEntry); },
      [this](edm::ProductID const& p) { return getIt(p); });
}

void BareRootProductGetterBase::getThinnedProducts(edm::ProductID const& pid,
                                                   std::vector<edm::WrapperBase const*>& foundContainers,
                                                   std::vector<unsigned int>& keys) const {
  Long_t eventEntry = branchMap_.getEventTree()->GetReadEntry();
  edm::detail::getThinnedProducts(
      pid,
      branchMap_.thinnedAssociationsHelper(),
      [this](edm::ProductID const& p) { return branchMap_.productToBranchID(p); },
      [this, eventEntry](edm::BranchID const& b) { return getThinnedAssociation(b, eventEntry); },
      [this](edm::ProductID const& p) { return getIt(p); },
      foundContainers,
      keys);
}

edm::OptionalThinnedKey BareRootProductGetterBase::getThinnedKeyFrom(edm::ProductID const& parentID,
                                                                     unsigned int key,
                                                                     edm::ProductID const& thinnedID) const {
  Long_t eventEntry = branchMap_.getEventTree()->GetReadEntry();
  edm::BranchID parent = branchMap_.productToBranchID(parentID);
  if (!parent.isValid())
    return std::monostate{};
  edm::BranchID thinned = branchMap_.productToBranchID(thinnedID);
  if (!thinned.isValid())
    return std::monostate{};
  try {
    auto ret = edm::detail::getThinnedKeyFrom_implementation(
        parentID,
        parent,
        key,
        thinnedID,
        thinned,
        branchMap_.thinnedAssociationsHelper(),
        [this, eventEntry](edm::BranchID const& branchID) { return getThinnedAssociation(branchID, eventEntry); });
    if (auto factory = std::get_if<edm::detail::GetThinnedKeyFromExceptionFactory>(&ret)) {
      return [func = *factory]() {
        auto ex = func();
        ex.addContext("Calling BareRootProductGetterBase::getThinnedKeyFrom()");
        return ex;
      };
    } else {
      return ret;
    }
  } catch (edm::Exception& ex) {
    ex.addContext("Calling BareRootProductGetterBase::getThinnedKeyFrom()");
    throw ex;
  }
}

BareRootProductGetterBase::Buffer* BareRootProductGetterBase::createNewBuffer(edm::BranchID const& branchID) const {
  //find the branch
  edm::ProductDescription const& bdesc = branchMap_.branchIDToBranch(branchID);

  TBranch* branch = branchMap_.getEventTree()->GetBranch(bdesc.branchName().c_str());
  if (nullptr == branch) {
    //we do not thrown on missing branches since 'getIt' should not throw under that condition
    return nullptr;
  }
  //find the class type
  std::string const fullName = edm::wrappedClassName(bdesc.className());
  edm::TypeWithDict classType(edm::TypeWithDict::byName(fullName));
  if (!bool(classType)) {
    throw cms::Exception("MissingDictionary") << "could not find dictionary for type '" << fullName << "'"
                                              << "\n Please make sure all the necessary libraries are available.";
    return nullptr;
  }

  TClass* rootClassType = TClass::GetClass(classType.typeInfo());
  if (nullptr == rootClassType) {
    throw cms::Exception("MissingRootDictionary") << "could not find a ROOT dictionary for type '" << fullName << "'"
                                                  << "\n Please make sure all the necessary libraries are available.";
    return nullptr;
  }
  void* address = rootClassType->New();

  static TClass const* edproductTClass = TClass::GetClass(typeid(edm::WrapperBase));
  edm::WrapperBase const* prod =
      static_cast<edm::WrapperBase const*>(rootClassType->DynamicCast(edproductTClass, address, true));
  if (nullptr == prod) {
    throw cms::Exception("FailedConversion") << "failed to convert a '" << fullName << "' to a edm::WrapperBase."
                                             << "Please contact developers since something is very wrong.";
  }

  //connect the instance to the branch
  //void* address  = wrapperObj.Address();
  Buffer b(prod, branch, address, rootClassType);
  idToBuffers_[branchID] = std::move(b);

  //As of 5.13 ROOT expects the memory address held by the pointer passed to
  // SetAddress to be valid forever
  address = &(idToBuffers_[branchID].address_);
  branch->SetAddress(address);

  return &(idToBuffers_[branchID]);
}

edm::ThinnedAssociation const* BareRootProductGetterBase::getThinnedAssociation(edm::BranchID const& branchID,
                                                                                Long_t eventEntry) const {
  edm::WrapperBase const* wrapperBase = getIt(branchID, eventEntry);
  if (wrapperBase == nullptr) {
    throw edm::Exception(edm::errors::LogicError)
        << "BareRootProductGetterBase::getThinnedAssociation, product ThinnedAssociation not found.\n";
  }
  if (!(typeid(edm::ThinnedAssociation) == wrapperBase->dynamicTypeInfo())) {
    throw edm::Exception(edm::errors::LogicError)
        << "BareRootProductGetterBase::getThinnedAssociation, product has wrong type, not a ThinnedAssociation.\n";
  }
  edm::Wrapper<edm::ThinnedAssociation> const* wrapper =
      static_cast<edm::Wrapper<edm::ThinnedAssociation> const*>(wrapperBase);

  edm::ThinnedAssociation const* thinnedAssociation = wrapper->product();
  return thinnedAssociation;
}
