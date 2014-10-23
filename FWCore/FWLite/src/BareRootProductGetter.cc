// -*- C++ -*-
//
// Package:     FWLite
// Class  :     BareRootProductGetter
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue May 23 11:03:31 EDT 2006
//

// user include files
#include "FWCore/FWLite/src/BareRootProductGetter.h"
#include "DataFormats/Common/interface/ThinnedAssociation.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

// system include files

#include "TROOT.h"
#include "TBranch.h"
#include "TClass.h"
#include "TFile.h"
#include "TTree.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
BareRootProductGetter::BareRootProductGetter() {
}

// BareRootProductGetter::BareRootProductGetter(BareRootProductGetter const& rhs) {
//    // do actual copying here;
// }

BareRootProductGetter::~BareRootProductGetter() {
}

//
// assignment operators
//
// BareRootProductGetter const& BareRootProductGetter::operator=(BareRootProductGetter const& rhs) {
//   //An exception safe implementation is
//   BareRootProductGetter temp(rhs);
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
edm::WrapperBase const*
BareRootProductGetter::getIt(edm::ProductID const& pid) const  {
  // std::cout << "getIt called " << pid << std::endl;
  TFile* currentFile = dynamic_cast<TFile*>(gROOT->GetListOfFiles()->Last());
  if(0 == currentFile) {
     throw cms::Exception("FileNotFound")
        << "unable to find the TFile '" << gROOT->GetListOfFiles()->Last() << "'\n"
        << "retrieved by calling 'gROOT->GetListOfFiles()->Last()'\n"
        << "Please check the list of files.";
  }
  if(branchMap_.updateFile(currentFile)) {
    idToBuffers_.clear();
  }
  TTree* eventTree = branchMap_.getEventTree();
  // std::cout << "eventTree " << eventTree << std::endl;
  if(0 == eventTree) {
     throw cms::Exception("NoEventsTree")
        << "unable to find the TTree '" << edm::poolNames::eventTreeName() << "' in the last open file, \n"
        << "file: '" << branchMap_.getFile()->GetName()
        << "'\n Please check that the file is a standard CMS ROOT format.\n"
        << "If the above is not the file you expect then please open your data file after all other files.";
  }
  Long_t eventEntry = eventTree->GetReadEntry();
  // std::cout << "eventEntry " << eventEntry << std::endl;
  branchMap_.updateEvent(eventEntry);
  if(eventEntry < 0) {
     throw cms::Exception("GetEntryNotCalled")
        << "please call GetEntry for the 'Events' TTree for each event in order to make edm::Ref's work."
        << "\n Also be sure to call 'SetAddress' for all Branches after calling the GetEntry."
        ;
  }

  edm::BranchID branchID = branchMap_.productToBranchID(pid);

  return getIt(branchID, eventEntry);
}

edm::WrapperBase const*
BareRootProductGetter::getIt(edm::BranchID const& branchID, Long_t eventEntry) const  {

  Buffer* buffer = nullptr;
  IdToBuffers::iterator itBuffer = idToBuffers_.find(branchID);

  // std::cout << "Buffers" << std::endl;
  if(itBuffer == idToBuffers_.end()) {
    buffer = createNewBuffer(branchID);
    // std::cout << "buffer " << buffer << std::endl;
    if(nullptr == buffer) {
       return nullptr;
    }
  } else {
    buffer = &(itBuffer->second);
  }
  if(nullptr == buffer) {
     throw cms::Exception("NullBuffer")
        << "Found a null buffer which is supposed to hold the data item."
        << "\n Please contact developers since this message should not happen.";
  }
  if(nullptr == buffer->branch_) {
     throw cms::Exception("NullBranch")
        << "The TBranch which should hold the data item is null."
        << "\n Please contact the developers since this message should not happen.";
  }
  if(buffer->eventEntry_ != eventEntry) {
    //NOTE: Need to reset address because user could have set the address themselves
    //std::cout << "new event" << std::endl;

    //ROOT WORKAROUND: Create new objects so any internal data cache will get cleared
    void* address = buffer->class_->New();

    static TClass const* edproductTClass = TClass::GetClass(typeid(edm::WrapperBase));
    edm::WrapperBase const* prod = static_cast<edm::WrapperBase const*>(buffer->class_->DynamicCast(edproductTClass,address,true));

    if(nullptr == prod) {
      cms::Exception("FailedConversion")
      << "failed to convert a '" << buffer->class_->GetName()
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
  if(!buffer->product_) {
     throw cms::Exception("BranchGetEntryFailed")
        << "Calling GetEntry with index " << eventEntry
        << "for branch " << buffer->branch_->GetName() << " failed.";
  }

  return buffer->product_.get();
}

edm::WrapperBase const*
BareRootProductGetter::getThinnedProduct(edm::ProductID const& pid, unsigned int& key) const {

  Long_t eventEntry = branchMap_.getEventTree()->GetReadEntry();
  edm::BranchID parent = branchMap_.productToBranchID(pid);
  if(!parent.isValid()) return nullptr;
  edm::ThinnedAssociationsHelper const& thinnedAssociationsHelper = branchMap_.thinnedAssociationsHelper();

  // Loop over thinned containers which were made by selecting elements from the parent container
  for(auto associatedBranches = thinnedAssociationsHelper.parentBegin(parent),
                         iEnd = thinnedAssociationsHelper.parentEnd(parent);
      associatedBranches != iEnd; ++associatedBranches) {

    edm::ThinnedAssociation const* thinnedAssociation =
      getThinnedAssociation(associatedBranches->association(), eventEntry);
    if(thinnedAssociation == nullptr) continue;

    if(associatedBranches->parent() != branchMap_.productToBranchID(thinnedAssociation->parentCollectionID())) {
      continue;
    }

    unsigned int thinnedIndex = 0;
    // Does this thinned container have the element referenced by key?
    // If yes, thinnedIndex is set to point to it in the thinned container
    if(!thinnedAssociation->hasParentIndex(key, thinnedIndex)) {
      continue;
    }
    // Get the thinned container and return a pointer if we can find it
    edm::ProductID const& thinnedCollectionPID = thinnedAssociation->thinnedCollectionID();
    edm::WrapperBase const* thinnedCollection = getIt(thinnedCollectionPID);
    if(thinnedCollection == nullptr) {
      // Thinned container is not found, try looking recursively in thinned containers
      // which were made by selecting elements from this thinned container.
      edm::WrapperBase const* thinnedFromRecursiveCall = getThinnedProduct(thinnedCollectionPID, thinnedIndex);
      if(thinnedFromRecursiveCall != nullptr) {
        key = thinnedIndex;
        return thinnedFromRecursiveCall;
      } else {
        continue;
      }
    }
    key = thinnedIndex;
    return thinnedCollection;
  }
  return nullptr;
}

void
BareRootProductGetter::getThinnedProducts(edm::ProductID const& pid,
                                          std::vector<edm::WrapperBase const*>& foundContainers,
                                          std::vector<unsigned int>& keys) const {

  Long_t eventEntry = branchMap_.getEventTree()->GetReadEntry();
  edm::BranchID parent = branchMap_.productToBranchID(pid);
  if(!parent.isValid()) return;
  edm::ThinnedAssociationsHelper const& thinnedAssociationsHelper = branchMap_.thinnedAssociationsHelper();

  // Loop over thinned containers which were made by selecting elements from the parent container
  for(auto associatedBranches = thinnedAssociationsHelper.parentBegin(parent),
                         iEnd = thinnedAssociationsHelper.parentEnd(parent);
      associatedBranches != iEnd; ++associatedBranches) {

    edm::ThinnedAssociation const* thinnedAssociation =
      getThinnedAssociation(associatedBranches->association(), eventEntry);
    if(thinnedAssociation == nullptr) continue;

    if(associatedBranches->parent() != branchMap_.productToBranchID(thinnedAssociation->parentCollectionID())) {
      continue;
    }

    unsigned int nKeys = keys.size();
    unsigned int doNotLookForThisIndex = std::numeric_limits<unsigned int>::max();
    std::vector<unsigned int> thinnedIndexes(nKeys, doNotLookForThisIndex);
    bool hasAny = false;
    for(unsigned k = 0; k < nKeys; ++k) {
      // Already found this one
      if(foundContainers[k] != nullptr) continue;
      // Already know this one is not in this thinned container
      if(keys[k] == doNotLookForThisIndex) continue;
      // Does the thinned container hold the entry of interest?
      // Modifies thinnedIndexes[k] only if it returns true and
      // sets it to the index in the thinned collection.
      if(thinnedAssociation->hasParentIndex(keys[k], thinnedIndexes[k])) {
        hasAny = true;
      }
    }
    if(!hasAny) {
      continue;
    }
    // Get the thinned container and set the pointers and indexes into
    // it (if we can find it)
    edm::ProductID thinnedCollectionPID = thinnedAssociation->thinnedCollectionID();
    edm::WrapperBase const* thinnedCollection = getIt(thinnedCollectionPID);

    if(thinnedCollection == nullptr) {
      // Thinned container is not found, try looking recursively in thinned containers
      // which were made by selecting elements from this thinned container.
      getThinnedProducts(thinnedCollectionPID, foundContainers, thinnedIndexes);
      for(unsigned k = 0; k < nKeys; ++k) {
        if(foundContainers[k] == nullptr) continue;
        if(thinnedIndexes[k] == doNotLookForThisIndex) continue;
        keys[k] = thinnedIndexes[k];
      }
    } else {
      for(unsigned k = 0; k < nKeys; ++k) {
        if(thinnedIndexes[k] == doNotLookForThisIndex) continue;
        keys[k] = thinnedIndexes[k];
        foundContainers[k] = thinnedCollection;
      }
    }
  }
}

BareRootProductGetter::Buffer*
BareRootProductGetter::createNewBuffer(edm::BranchID const& branchID) const {
  //find the branch
  edm::BranchDescription const& bdesc = branchMap_.branchIDToBranch(branchID);

  TBranch* branch= branchMap_.getEventTree()->GetBranch(bdesc.branchName().c_str());
  if(nullptr == branch) {
     //we do not thrown on missing branches since 'getIt' should not throw under that condition
    return nullptr;
  }
  //find the class type
  std::string const fullName = edm::wrappedClassName(bdesc.className());
  edm::TypeWithDict classType(edm::TypeWithDict::byName(fullName));
  if(!bool(classType)) {
    throw cms::Exception("MissingDictionary")
       << "could not find dictionary for type '" << fullName << "'"
       << "\n Please make sure all the necessary libraries are available.";
    return nullptr;
  }

  TClass* rootClassType = TClass::GetClass(classType.typeInfo());
  if(nullptr == rootClassType) {
    throw cms::Exception("MissingRootDictionary")
    << "could not find a ROOT dictionary for type '" << fullName << "'"
    << "\n Please make sure all the necessary libraries are available.";
    return nullptr;
  }
  void* address = rootClassType->New();

  static TClass const* edproductTClass = TClass::GetClass(typeid(edm::WrapperBase));
  edm::WrapperBase const* prod = static_cast<edm::WrapperBase const*>( rootClassType->DynamicCast(edproductTClass,address,true));
  if(nullptr == prod) {
     throw cms::Exception("FailedConversion")
        << "failed to convert a '" << fullName
        << "' to a edm::WrapperBase."
        << "Please contact developers since something is very wrong.";
  }

  //connect the instance to the branch
  //void* address  = wrapperObj.Address();
  Buffer b(prod, branch, address, rootClassType);
  idToBuffers_[branchID] = b;

  //As of 5.13 ROOT expects the memory address held by the pointer passed to
  // SetAddress to be valid forever
  address = &(idToBuffers_[branchID].address_);
  branch->SetAddress(address);

  return &(idToBuffers_[branchID]);
}

edm::ThinnedAssociation const*
BareRootProductGetter::getThinnedAssociation(edm::BranchID const& branchID, Long_t eventEntry) const {

  edm::WrapperBase const* wrapperBase = getIt(branchID, eventEntry);
  if(wrapperBase == nullptr) {
    throw edm::Exception(edm::errors::LogicError)
      << "BareRootProductGetter::getThinnedAssociation, product ThinnedAssociation not found.\n";
  }
  if(!(typeid(edm::ThinnedAssociation) == wrapperBase->dynamicTypeInfo())) {
    throw edm::Exception(edm::errors::LogicError)
      << "BareRootProductGetter::getThinnedAssociation, product has wrong type, not a ThinnedAssociation.\n";
  }
  edm::Wrapper<edm::ThinnedAssociation> const* wrapper =
    static_cast<edm::Wrapper<edm::ThinnedAssociation> const*>(wrapperBase);

  edm::ThinnedAssociation const* thinnedAssociation = wrapper->product();
  return thinnedAssociation;
}
