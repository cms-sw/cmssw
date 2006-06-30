// -*- C++ -*-
//
// Package:     FWLite
// Class  :     TFWLiteSelector
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Jun 27 17:58:10 EDT 2006
// $Id$
//

// system include files
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TBranch.h"
#include "Reflex/Type.h"
#include "Reflex/Object.h"
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/TFWLiteSelector/interface/TFWLiteSelector.h"

#include "IOPool/Common/interface/PoolNames.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "DataFormats/Common/interface/ProcessNameList.h"
#include "DataFormats/Common/interface/EventAux.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Common/interface/EventProvenance.h"

namespace edm {
  namespace root {
    class FWLiteDelayedReader : public DelayedReader {
     public:
      virtual std::auto_ptr<EDProduct> get(BranchKey const& k, EDProductGetter const* ep) const;
      void setEntry(Long64_t iEntry) { entry_ = iEntry; }
      void setTree(TTree* iTree) {eventTree_ = iTree;}
      void set(const edm::ProductRegistry* iReg) { reg_ = iReg;}
     private:
      Long64_t entry_;
      TTree* eventTree_;
      const edm::ProductRegistry* reg_;
    };
    
    std::auto_ptr<EDProduct> 
    FWLiteDelayedReader::get(BranchKey const& k, EDProductGetter const* ep) const
    {
      edm::ProductRegistry::ProductList::const_iterator itFind= reg_->productList().find(k);
      if(itFind == reg_->productList().end()) {
        throw edm::Exception(edm::errors::ProductNotFound)<<"could not find entry for product "<<k;
      }
      const edm::BranchDescription& bDesc = itFind->second;
      
      TBranch* branch= eventTree_->GetBranch( bDesc.branchName().c_str() );
      if( 0 == branch) {
        throw cms::Exception("MissingBranch") 
        <<"could not find branch named '"<<bDesc.branchName()<<"'"
        <<"\n Perhaps the data being requested was not saved in this file?";
      }
      //find the class type
      const std::string fullName = edm::wrappedClassName(bDesc.className());
      ROOT::Reflex::Type classType = ROOT::Reflex::Type::ByName(fullName);
      if( classType == ROOT::Reflex::Type() ) {
        throw cms::Exception("MissingDictionary") 
        <<"could not find dictionary for type '"<<fullName<<"'"
        <<"\n Please make sure all the necessary libraries are available.";
        return std::auto_ptr<EDProduct>();
      }
      
      //use reflex to create an instance of it
      ROOT::Reflex::Object wrapperObj = classType.Construct();
      if( 0 == wrapperObj.Address() ) {
        throw cms::Exception("FailedToCreate") <<"could not create an instance of '"<<fullName<<"'";
      }
      void* address  = wrapperObj.Address();
      branch->SetAddress( &address );
      
      ROOT::Reflex::Object edProdObj = wrapperObj.CastObject( ROOT::Reflex::Type::ByName("edm::EDProduct") );
      
      edm::EDProduct* prod = reinterpret_cast<edm::EDProduct*>(edProdObj.Address());
      if(0 == prod) {
        throw cms::Exception("FailedConversion")
	<<"failed to convert a '"<<fullName
	<<"' to a edm::EDProduct."
	<<"Please contact developers since something is very wrong.";
      }
      branch->GetEntry(entry_);
      return std::auto_ptr<EDProduct>(prod);
    }
    
    struct TFWLiteSelectorMembers {
      TFWLiteSelectorMembers():
      tree_(0), reader_(new FWLiteDelayedReader) {
        reader_->set(&reg_);}
      void setTree( TTree* iTree) {
        tree_ = iTree;
        reader_->setTree(iTree);
      }
      TTree* tree_;
      ProductRegistry reg_;
      ProcessNameList processNames_;
      boost::shared_ptr<FWLiteDelayedReader> reader_;
      typedef std::map<ProductID, BranchDescription> ProductMap;
      ProductMap productMap_;
    };
  }
}
  

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TFWLiteSelector::TFWLiteSelector() : m_(new edm::root::TFWLiteSelectorMembers),
				     everythingOK_(false)
{
}

// TFWLiteSelector::TFWLiteSelector(const TFWLiteSelector& rhs)
// {
//    // do actual copying here;
// }

TFWLiteSelector::~TFWLiteSelector()
{
  delete m_;
}

//
// assignment operators
//
// const TFWLiteSelector& TFWLiteSelector::operator=(const TFWLiteSelector& rhs)
// {
//   //An exception safe implementation is
//   TFWLiteSelector temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
TFWLiteSelector::Begin(TTree * iTree) { 
  Init(iTree);
  begin();
}

void
TFWLiteSelector::SlaveBegin(TTree *iTree) { 
  Init(iTree);
  preProcessing(*fOutput);
}

void
TFWLiteSelector::Init(TTree *iTree) { 
  if(iTree==0) return;
  m_->setTree(iTree);
}


Bool_t
TFWLiteSelector::Notify() { 
   std::cout <<"Notify start"<<std::endl;
  //we have switched to a new file  
  //get new file from Tree
  if(0==m_->tree_) {
    std::cout <<"No tree"<<std::endl;
    return kFALSE;
  }
  TFile* file = m_->tree_->GetCurrentFile();
  if(0 == file) {
     //When in Rome, do as the Romans
     TChain* chain = dynamic_cast<TChain*>(m_->tree_);
     if(0 == chain) {
	std::cout <<"No file"<<std::endl;
	return kFALSE;
     }
     file = chain->GetFile();
     if(0==file) {
	std::cout <<"No file"<<std::endl;
	return kFALSE;
     }
  }
  //look up meta-data
  //get product registry
  edm::ProductRegistry* pReg = &(m_->reg_);
  TTree* metaDataTree = dynamic_cast<TTree*>(file->Get(edm::poolNames::metaDataTreeName().c_str()) );
  if ( 0 != metaDataTree) {
    metaDataTree->SetBranchAddress(edm::poolNames::productDescriptionBranchName().c_str(), &(pReg) );
    metaDataTree->GetEntry(0);
    m_->reg_.setFrozen();
  }
  m_->productMap_.erase(m_->productMap_.begin(),m_->productMap_.end());
  edm::ProductRegistry::ProductList const& prodList = pReg->productList();
  for (edm::ProductRegistry::ProductList::const_iterator it = prodList.begin();
       it != prodList.end(); ++it) {
     edm::BranchDescription const& prod = it->second;
     prod.init();
     m_->productMap_.insert(std::make_pair(it->second.productID_, it->second));
  }  
  std::cout <<"Notify end"<<std::endl;
  everythingOK_ = true;
  return kTRUE; 
}

Bool_t
TFWLiteSelector::Process(Long64_t iEntry) { 
   std::cout <<"Process start"<<std::endl;
   if(everythingOK_) {
      edm::EventAux aux;
      edm::EventAux* pAux= &aux;
      TBranch* branch = m_->tree_->GetBranch("EventAux");

      edm::EventProvenance prov;
      edm::EventProvenance* pProv=&prov;
      TBranch* provBranch = m_->tree_->GetBranch("Provenance");

      branch->SetAddress(&pAux);
      provBranch->SetAddress(&pProv);
      branch->GetEntry(iEntry);
      provBranch->GetEntry(iEntry);

//NEW      m_->processNames_ = aux.processHistory();
      m_->processNames_ = aux.process_history_;

//      std::cout <<"ProcessNames\n";
//      for(edm::ProcessNameList::const_iterator itName = m_->processNames_.begin();
//	  itName != m_->processNames_.end();
//	  ++itName) {
//	 std::cout <<"  "<<*itName<< std::endl;
      //     }

      try {
	 m_->reader_->setEntry(iEntry);
//NEW	 edm::EventPrincipal ep(aux.id(), aux.time(),m_->reg_, m_->processNames_, m_->reader_);
	 edm::EventPrincipal ep(aux.id_, aux.time_,m_->reg_, m_->processNames_, m_->reader_);

	 using namespace edm;
	 std::vector<BranchEntryDescription>::iterator pit = prov.data_.begin();
	 std::vector<BranchEntryDescription>::iterator pitEnd = prov.data_.end();
	 for (; pit != pitEnd; ++pit) {
	    if (pit->status != BranchEntryDescription::Success) continue;
	    std::auto_ptr<Provenance> prov(new Provenance);
	    prov->event = *pit;
	    prov->product = m_->productMap_[prov->event.productID_];
	    std::auto_ptr<Group> g(new Group(prov));
	    ep.addGroup(g);
	 }

	 edm::ModuleDescription md;
	 edm::Event event(ep,md);
	 
	 //Make the event principal accessible to edm::Ref's
	 edm::EDProductGetter::Operate sentry(&ep);
	 process(event);
      } catch( const std::exception& iEx ) {
	 std::cout <<"While processing entry "<<iEntry<<" the following exception was caught \n"
		   <<iEx.what()<<std::endl;
      } catch(...) {
	 std::cout <<"While processing entry "<<iEntry<<" an unknown exception was caught" << std::endl;
      }
   }
   std::cout <<"Process end"<<std::endl;
  return kFALSE; 
}

void
TFWLiteSelector::SlaveTerminate() { 
  postProcessing();
}

void
TFWLiteSelector::Terminate() {
  terminate(*fOutput);
}

//
// const member functions
//

//
// static member functions
//
