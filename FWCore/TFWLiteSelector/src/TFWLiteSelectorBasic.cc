// -*- C++ -*-
//
// Package:     TFWLiteSelector
// Class  :     TFWLiteSelectorBasic
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Jun 27 17:58:10 EDT 2006
// $Id: TFWLiteSelectorBasic.cc,v 1.42 2008/11/28 17:44:31 wmtan Exp $
//

// system include files
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TBranch.h"
#include "TClass.h"
#include "Reflex/Type.h"
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/TFWLiteSelector/interface/TFWLiteSelectorBasic.h"

#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventEntryDescription.h" // kludge to allow compilation
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/DelayedReader.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
namespace edm {
  namespace root {
    class FWLiteDelayedReader : public DelayedReader {
     public:
      FWLiteDelayedReader(): entry_(-1),eventTree_(0),reg_() {}
      void setEntry(Long64_t iEntry) { entry_ = iEntry; }
      void setTree(TTree* iTree) {eventTree_ = iTree;}
      void set(boost::shared_ptr<const edm::ProductRegistry> iReg) { reg_ = iReg;}
     private:
      virtual std::auto_ptr<EDProduct> getProduct_(BranchKey const& k, EDProductGetter const* ep) const;
      virtual std::auto_ptr<EventEntryDescription> getProvenance_(BranchKey const&) const {
        return std::auto_ptr<EventEntryDescription>();
      }
      Long64_t entry_;
      TTree* eventTree_;
      boost::shared_ptr<const edm::ProductRegistry>(reg_);
    };
    
    std::auto_ptr<EDProduct> 
    FWLiteDelayedReader::getProduct_(BranchKey const& k, EDProductGetter const* ep) const
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
      Reflex::Type classType = Reflex::Type::ByName(fullName);
      if( classType == Reflex::Type() ) {
        throw cms::Exception("MissingDictionary") 
        <<"could not find dictionary for type '"<<fullName<<"'"
        <<"\n Please make sure all the necessary libraries are available.";
        return std::auto_ptr<EDProduct>();
      }
      
      //We can't use reflex to create the instance since Reflex uses 'malloc' instead of new
      /*
      //use reflex to create an instance of it
      Reflex::Object wrapperObj = classType.Construct();
      if( 0 == wrapperObj.Address() ) {
        throw cms::Exception("FailedToCreate") <<"could not create an instance of '"<<fullName<<"'";
      }
      void* address  = wrapperObj.Address();
       */
      TClass* rootClassType=TClass::GetClass(classType.TypeInfo());
      if( 0 == rootClassType) {
        throw cms::Exception("MissingRootDictionary")
        <<"could not find a ROOT dictionary for type '"<<fullName<<"'"
        <<"\n Please make sure all the necessary libraries are available.";
        return std::auto_ptr<EDProduct>();
      }
      void* address = rootClassType->New();
      branch->SetAddress( &address );
      
      /*
      Reflex::Object edProdObj = wrapperObj.CastObject( Reflex::Type::ByName("edm::EDProduct") );
      
      edm::EDProduct* prod = reinterpret_cast<edm::EDProduct*>(edProdObj.Address());
       */
      static TClass* edproductTClass = TClass::GetClass( typeid(edm::EDProduct)); 
      edm::EDProduct* prod = reinterpret_cast<edm::EDProduct*>( rootClassType->DynamicCast(edproductTClass,address,true));
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
      tree_(0),
      metaTree_(0),
      reg_(new edm::ProductRegistry()),
      processNames_(),
      reader_(new FWLiteDelayedReader),
      productMap_(),
      prov_(),
      pointerToBranchBuffer_()
      {
        reader_->set(reg_);}
      void setTree( TTree* iTree) {
        tree_ = iTree;
        reader_->setTree(iTree);
      }
      void setMetaTree( TTree* iTree) {
        metaTree_ = iTree;
      }
      TTree* tree_;
      TTree* metaTree_;
      TTree* eventHistoryTree_;
      boost::shared_ptr<ProductRegistry> reg_;
      ProcessHistory processNames_;
      boost::shared_ptr<FWLiteDelayedReader> reader_;
      typedef std::map<ProductID, BranchDescription> ProductMap;
      ProductMap productMap_;
      std::vector<edm::EventEntryDescription> prov_;
      std::vector<edm::EventEntryDescription*> pointerToBranchBuffer_;
      edm::FileFormatVersion fileFormatVersion_;
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
TFWLiteSelectorBasic::TFWLiteSelectorBasic() : m_(new edm::root::TFWLiteSelectorMembers),
				     everythingOK_(false)
{
}

// TFWLiteSelectorBasic::TFWLiteSelectorBasic(const TFWLiteSelectorBasic& rhs)
// {
//    // do actual copying here;
// }

TFWLiteSelectorBasic::~TFWLiteSelectorBasic()
{
  delete m_;
}

//
// assignment operators
//
// const TFWLiteSelectorBasic& TFWLiteSelectorBasic::operator=(const TFWLiteSelectorBasic& rhs)
// {
//   //An exception safe implementation is
//   TFWLiteSelectorBasic temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
TFWLiteSelectorBasic::Begin(TTree * iTree) { 
  Init(iTree);
  begin(fInput);
}

void
TFWLiteSelectorBasic::SlaveBegin(TTree *iTree) { 
  Init(iTree);
  preProcessing(fInput,*fOutput);
}

void
TFWLiteSelectorBasic::Init(TTree *iTree) { 
  if(iTree==0) return;
  m_->setTree(iTree);
}


Bool_t
TFWLiteSelectorBasic::Notify() { 
   //std::cout <<"Notify start"<<std::endl;
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
  setupNewFile(*file);
  return everythingOK_ ? kTRUE: kFALSE; 
}

Bool_t
TFWLiteSelectorBasic::Process(Long64_t iEntry) { 
   //std::cout <<"Process start"<<std::endl;
   if(everythingOK_) {
      edm::EventAuxiliary aux;
      edm::EventAuxiliary* pAux= &aux;
      TBranch* branch = m_->tree_->GetBranch(edm::BranchTypeToAuxiliaryBranchName(edm::InEvent).c_str());

      branch->SetAddress(&pAux);
      //provBranch->SetAddress(&pProv);
      branch->GetEntry(iEntry);
      //provBranch->GetEntry(iEntry);
      //CDJ turn off reading meta tree until we fix handling of provenance
      //m_->metaTree_->GetEntry(iEntry);

//NEW      m_->processNames_ = aux.processHistory();

//      std::cout <<"ProcessNames\n";
//      for(edm::ProcessNameList::const_iterator itName = m_->processNames_.begin(),
//        itNameEnd = m_->processNames_.end();
//	  itName != itNameEnd;
//	  ++itName) {
//	 std::cout <<"  "<<*itName<< std::endl;
      //     }

      edm::History history;
      if (m_->fileFormatVersion_.value_ >= 7) {
         edm::History* pHistory = &history;
         TBranch* eventHistoryBranch = m_->eventHistoryTree_->GetBranch(edm::poolNames::eventHistoryBranchName().c_str());
         if (!eventHistoryBranch)
            throw edm::Exception(edm::errors::FatalRootError)
            << "Failed to find history branch in event history tree";
         eventHistoryBranch->SetAddress(&pHistory);
         m_->eventHistoryTree_->GetEntry(iEntry);
         aux.processHistoryID_ = history.processHistoryID();
      }
/*
      try {
	 m_->reader_->setEntry(iEntry);
	 edm::ProcessConfiguration pc;
	 boost::shared_ptr<edm::ProductRegistry const> reg(m_->reg_);
	 edm::RunAuxiliary runAux(aux.run(), aux.time(), aux.time());
	 boost::shared_ptr<edm::RunPrincipal> rp(new edm::RunPrincipal(runAux, reg, pc));
	 edm::LuminosityBlockAuxiliary lumiAux(rp->run(), 1, aux.time(), aux.time());
	 boost::shared_ptr<edm::LuminosityBlockPrincipal>lbp(
	    new edm::LuminosityBlockPrincipal(lumiAux, reg, pc));
	 lbp->setRunPrincipal(rp);
	 boost::shared_ptr<edm::BranchMapper> mapper(new edm::BranchMapper);
	 edm::EventPrincipal ep(aux, reg, pc, aux.processHistoryID(), mapper, m_->reader_);
	 ep.setLuminosityBlockPrincipal(lbp);
         ep.setHistory(history);
         m_->processNames_ = ep.processHistory();

	 using namespace edm;
	 std::map<ProductID, BranchDescription>::iterator pit = m_->productMap_.begin();
	 std::map<ProductID, BranchDescription>::iterator pitEnd = m_->productMap_.end();
	 for (; pit != pitEnd; ++pit) {
	    BranchDescription &product = pit->second;
            if (not product.oldProductID().isValid()) continue;
	    ep.addGroup(edm::ConstBranchDescription(product));
	 }

	 edm::ModuleDescription md;
	 edm::Event event(ep,md);
	 
	 //Make the event principal accessible to edm::Ref's
	 edm::EDProductGetter::Operate sentry(ep.prodGetter());
	 process(event);
      } catch( const std::exception& iEx ) {
	 std::cout <<"While processing entry "<<iEntry<<" the following exception was caught \n"
		   <<iEx.what()<<std::endl;
      } catch(...) {
	 std::cout <<"While processing entry "<<iEntry<<" an unknown exception was caught" << std::endl;
      }
*/
   }
   //std::cout <<"Process end"<<std::endl;
  return kFALSE; 
}

void
TFWLiteSelectorBasic::SlaveTerminate() { 
  postProcessing(*fOutput);
}

void
TFWLiteSelectorBasic::Terminate() {
  terminate(*fOutput);
}

void
TFWLiteSelectorBasic::setupNewFile(TFile& iFile) { 
  //look up meta-data
  //get product registry
  edm::ProductRegistry* pReg = &(*m_->reg_);
  typedef std::map<edm::ParameterSetID, edm::ParameterSetBlob> PsetMap;
  PsetMap psetMap;
  edm::ProcessHistoryMap pHistMap;
  edm::ModuleDescriptionMap mdMap;
  PsetMap *psetMapPtr = &psetMap;
  edm::ProcessHistoryMap *pHistMapPtr = &pHistMap;
  edm::ModuleDescriptionMap *mdMapPtr = &mdMap;
  edm::FileFormatVersion* fftPtr = &(m_->fileFormatVersion_);
   
  TTree* metaDataTree = dynamic_cast<TTree*>(iFile.Get(edm::poolNames::metaDataTreeName().c_str()) );
  if ( 0 != metaDataTree) {
    metaDataTree->SetBranchAddress(edm::poolNames::productDescriptionBranchName().c_str(), &(pReg) );
    metaDataTree->SetBranchAddress(edm::poolNames::parameterSetMapBranchName().c_str(), &psetMapPtr);
    metaDataTree->SetBranchAddress(edm::poolNames::processHistoryMapBranchName().c_str(), &pHistMapPtr);
    // metaDataTree->SetBranchAddress(edm::poolNames::moduleDescriptionMapBranchName().c_str(), &mdMapPtr); // kludge to allow compilation
    metaDataTree->SetBranchAddress(edm::poolNames::fileFormatVersionBranchName().c_str(), &fftPtr);
    metaDataTree->GetEntry(0);
    m_->reg_->setFrozen();
  } else {
    std::cout <<"could not find TTree "<<edm::poolNames::metaDataTreeName() <<std::endl;
    everythingOK_ = false;
    return;
  }
  m_->metaTree_ = dynamic_cast<TTree*>(iFile.Get(edm::poolNames::eventMetaDataTreeName().c_str()));
  if( 0 == m_->metaTree_) {
    std::cout <<"could not find TTree "<<edm::poolNames::eventMetaDataTreeName() <<std::endl;
    everythingOK_ = false;
    return;
  }

  // Merge into the registries. For now, we do NOT merge the product registry.
  edm::pset::Registry& psetRegistry = *edm::pset::Registry::instance();
  for (PsetMap::const_iterator i = psetMap.begin(), iEnd = psetMap.end();
      i != iEnd; ++i) {
    psetRegistry.insertMapped(edm::ParameterSet(i->second.pset_));
  } 
  edm::ProcessHistoryRegistry & processNameListRegistry = *edm::ProcessHistoryRegistry::instance();
  for (edm::ProcessHistoryMap::const_iterator j = pHistMap.begin(), jEnd = pHistMap.end();
      j != jEnd; ++j) {
    processNameListRegistry.insertMapped(j->second);
  } 
  edm::ModuleDescriptionRegistry & moduleDescriptionRegistry = *edm::ModuleDescriptionRegistry::instance();
  for (edm::ModuleDescriptionMap::const_iterator k = mdMap.begin(), kEnd = mdMap.end();
      k != kEnd; ++k) {
    moduleDescriptionRegistry.insertMapped(k->second);
  } 
  
  m_->productMap_.erase(m_->productMap_.begin(),m_->productMap_.end());
  m_->pointerToBranchBuffer_.erase(m_->pointerToBranchBuffer_.begin(),
                                   m_->pointerToBranchBuffer_.end());
  edm::ProductRegistry::ProductList const& prodList = pReg->productList();
  std::vector<edm::EventEntryDescription> temp( prodList.size(), edm::EventEntryDescription() );
  m_->prov_.swap( temp);
  std::vector<edm::EventEntryDescription>::iterator itB = m_->prov_.begin();
  m_->pointerToBranchBuffer_.reserve(prodList.size());
  for (edm::ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end();
       it != itEnd; ++it, ++itB) {
    edm::BranchDescription const& prod = it->second;
    if( prod.branchType() == edm::InEvent) {
      prod.init();
      //NEED to do this and check to see if branch exists
      //prod.present_ = (branch != 0);
      m_->productMap_.insert(std::make_pair(it->second.oldProductID(), it->second));
      //std::cout <<"id "<<it->second.oldProductID()<<" branch "<<it->second.branchName()<<std::endl;
      m_->pointerToBranchBuffer_.push_back( & (*itB));
      void* tmp = &(m_->pointerToBranchBuffer_.back());
      //edm::EventEntryDescription* tmp = & (*itB);
      //CDJ need to fix provenance and be backwards compatible, for now just don't read the branch
      //m_->metaTree_->SetBranchAddress( prod.branchName().c_str(), tmp);
    }
  }  
  //std::cout <<"Notify end"<<std::endl;
   
   if (m_->fileFormatVersion_.value_ >= 7) {
      m_->eventHistoryTree_ = dynamic_cast<TTree*>(iFile.Get(edm::poolNames::eventHistoryTreeName().c_str()));
      if(0==m_->eventHistoryTree_) {
         std::cout <<"could not find TTree "<<edm::poolNames::eventHistoryTreeName() <<std::endl;
         everythingOK_ = false;
         return;
      }
   }
  everythingOK_ = true;
}

//
// const member functions
//

//
// static member functions
//
