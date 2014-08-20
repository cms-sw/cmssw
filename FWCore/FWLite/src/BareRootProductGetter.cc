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

#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

// system include files

#include "TROOT.h"
#include "TBranch.h"
#include "TClass.h"
#include "TFile.h"
#include "TTree.h"

#include <memory>
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
edm::EDProduct const*
BareRootProductGetter::getIt(edm::ProductID const& iID) const  {
  // std::cout << "getIt called" << std::endl;
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

  Buffer* buffer = nullptr;
  IdToBuffers::iterator itBuffer = idToBuffers_.find(iID);
  // std::cout << "Buffers" << std::endl;
  if(itBuffer == idToBuffers_.end()) {
    buffer = createNewBuffer(iID);
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

    static TClass const* edproductTClass = TClass::GetClass(typeid(edm::EDProduct));
    edm::EDProduct const* prod = static_cast<edm::EDProduct const*>(buffer->class_->DynamicCast(edproductTClass,address,true));

    if(nullptr == prod) {
      cms::Exception("FailedConversion")
      << "failed to convert a '" << buffer->class_->GetName()
      << "' to a edm::EDProduct."
      << "Please contact developers since something is very wrong.";
    }
    buffer->address_ = address;
    buffer->product_ = std::shared_ptr<edm::EDProduct const>(prod);
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

BareRootProductGetter::Buffer*
BareRootProductGetter::createNewBuffer(edm::ProductID const& iID) const {
  //find the branch
  edm::BranchDescription bdesc = branchMap_.productToBranch(iID);

  TBranch* branch= branchMap_.getEventTree()->GetBranch(bdesc.branchName().c_str());
  if(nullptr == branch) {
     //we do not thrown on missing branches since 'getIt' should not throw under that condition
    return nullptr;
  }
  //find the class type
  std::string const fullName = edm::wrappedClassName(bdesc.className());
  edm::TypeWithDict classType(edm::TypeWithDict::byName(fullName));
  if(!bool(classType)) {
    cms::Exception("MissingDictionary")
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

  static TClass const* edproductTClass = TClass::GetClass(typeid(edm::EDProduct));
  edm::EDProduct const* prod = reinterpret_cast<edm::EDProduct const*>( rootClassType->DynamicCast(edproductTClass,address,true));
  if(nullptr == prod) {
     cms::Exception("FailedConversion")
        << "failed to convert a '" << fullName
        << "' to a edm::EDProduct."
        << "Please contact developers since something is very wrong.";
  }

  //connect the instance to the branch
  //void* address  = wrapperObj.Address();
  Buffer b(prod, branch, address, rootClassType);
  idToBuffers_[iID] = b;

  //As of 5.13 ROOT expects the memory address held by the pointer passed to
  // SetAddress to be valid forever
  address = &(idToBuffers_[iID].address_);
  branch->SetAddress(address);

  return &(idToBuffers_[iID]);
}

//
// static member functions
//
