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
// $Id$
//

// system include files

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "Reflex/Type.h"
#include "Reflex/Object.h"

// user include files
#include "FWCore/FWLite/src/BareRootProductGetter.h"
#include "IOPool/Common/interface/PoolNames.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/EDProduct.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
BareRootProductGetter::BareRootProductGetter():
presentFile_(0),
eventTree_(0),
eventEntry_(-1)
{
}

// BareRootProductGetter::BareRootProductGetter(const BareRootProductGetter& rhs)
// {
//    // do actual copying here;
// }

BareRootProductGetter::~BareRootProductGetter()
{
}

//
// assignment operators
//
// const BareRootProductGetter& BareRootProductGetter::operator=(const BareRootProductGetter& rhs)
// {
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
  if(gFile == 0 ) {
    return 0;
  }
  if(gFile !=presentFile_) {
    setupNewFile(gFile);
  } else {
    //could still have a new TFile which just happens to share the same memory address as the previous file
    //will assume that if the Event tree's address and entry are the same as before then we do not have
    // to treat this like a new file
    TTree* eventTreeTemp = dynamic_cast<TTree*>(gFile->Get(edm::poolNames::eventTreeName().c_str()));
    if(eventTreeTemp != eventTree_ ||
       eventTree_->GetReadEntry() != eventEntry_) {
      setupNewFile(gFile);
    }
  }
  if (0 == eventTree_) {
    return 0;
  }
  Buffer* buffer = 0;
  IdToBuffers::iterator itBuffer = idToBuffers_.find(iID);
  if( itBuffer == idToBuffers_.end() ) {
    buffer = createNewBuffer(iID);
    if( 0 == buffer ) {
      return 0;
    }
  } else {
    buffer = &(itBuffer->second);
  }
  eventEntry_ = eventTree_->GetReadEntry();  
  if( eventEntry_ < 0 ) {
    std::cout <<"please call GetEntry for the 'Events' TTree in order to make edm::Ref's work"<<std::endl;
    return 0;
  }
  
  if(0==buffer) {
    return 0;
  }
  if(0==buffer->branch_) {
    return 0;
  }
  buffer->branch_->GetEntry( eventEntry_ );
  
  return buffer->product_.get();
}

void
BareRootProductGetter::setupNewFile(TFile* iFile) const
{
  presentFile_ = iFile;
  eventTree_= dynamic_cast<TTree*>(iFile->Get(edm::poolNames::eventTreeName().c_str()));
  
  if(0!= eventTree_) {
    //get product registry
    edm::ProductRegistry reg;
    edm::ProductRegistry* pReg = &reg;
    TTree* metaDataTree = dynamic_cast<TTree*>(iFile->Get(edm::poolNames::metaDataTreeName().c_str()) );
    if ( 0 != metaDataTree) {
      metaDataTree->SetBranchAddress(edm::poolNames::productDescriptionBranchName().c_str(), &(pReg) );
      metaDataTree->GetEntry(0);
      reg.setFrozen();
      
      IdToBranchDesc temp;
      idToBranchDesc_.swap(temp);
      IdToBuffers temp2;
      idToBuffers_.swap(temp2);
      const edm::ProductRegistry::ProductList& prodList = reg.productList();
      for(edm::ProductRegistry::ProductList::const_iterator itProd = prodList.begin();
          itProd != prodList.end();
          ++itProd) {
        //this has to be called since 'branchName' is not stored and the 'init' method is supposed to
        // regenerate it
        itProd->second.init();
        idToBranchDesc_.insert(IdToBranchDesc::value_type(itProd->second.productID(), itProd->second) );
      }
    }
  }
  eventEntry_ = -1;
}

BareRootProductGetter::Buffer* 
BareRootProductGetter::createNewBuffer(const edm::ProductID& iID) const
{
  //find the branch
  IdToBranchDesc::iterator itBD = idToBranchDesc_.find(iID);
  if( itBD == idToBranchDesc_.end() ) {
    std::cout <<"could not find product ID "<<iID<<std::endl;
    return 0;
  }
  
  TBranch* branch= eventTree_->GetBranch( itBD->second.branchName().c_str() );
  if( 0 == branch) {
    std::cout <<"could not find branch named '"<<itBD->second.branchName()<<"'"<<std::endl;
    return 0;
  }
  //find the class type
  static const std::string wrapperBegin("edm::Wrapper<");
  static const std::string wrapperEnd1(">");
  static const std::string wrapperEnd2(" >");
  const std::string& fullName = itBD->second.className();
  ROOT::Reflex::Type classType = ROOT::Reflex::Type::ByName( wrapperBegin + fullName
                                                             + (fullName[fullName.size()-1]=='>' ? wrapperEnd2 : wrapperEnd1) );
  if( classType == ROOT::Reflex::Type() ) {
    std::cout <<"could not find Reflex Type '"<<fullName<<"'"<<std::endl;
    return 0;
  }
  
  //use reflex to create an instance of it
  ROOT::Reflex::Object wrapperObj = classType.Construct();
  if( 0 == wrapperObj.Address() ) {
    std::cout <<"could not create in instance of '"<<fullName<<"'"<<std::endl;
    return 0;
  }
  void* address  = wrapperObj.Address();
  branch->SetAddress( &address );
      
  ROOT::Reflex::Object edProdObj = wrapperObj.CastObject( ROOT::Reflex::Type::ByName("edm::EDProduct") );
  
  edm::EDProduct* prod = reinterpret_cast<edm::EDProduct*>(edProdObj.Address());
  
  //connect the instance to the branch
  Buffer b(prod, branch);
  idToBuffers_[iID]=b;
  
  return &(idToBuffers_[iID]);
}

//
// static member functions
//
