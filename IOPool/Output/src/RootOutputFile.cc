// $Id: RootOutputFile.cc,v 1.12 2007/09/10 20:27:08 wmtan Exp $

#include "RootOutputFile.h"
#include "PoolOutputModule.h"

#include "DataFormats/Provenance/interface/EventAuxiliary.h" 
#include "IOPool/Common/interface/RootChains.h"

#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "FWCore/Utilities/interface/GetFileFormatVersion.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "POOLCore/Guid.h"

#include "TTree.h"
#include "TFile.h"
#include "Rtypes.h"

#include <map>
#include <iomanip>

namespace edm {
  RootOutputFile::RootOutputFile(PoolOutputModule *om, std::string const& fileName, std::string const& logicalFileName) :
      chains_(om->wantAllEvents() ? RootChains::instance() : RootChains()),
      outputItemList_(), 
      file_(fileName),
      logicalFile_(logicalFileName),
      reportToken_(0),
      eventCount_(0),
      fileSizeCheckEvent_(100),
      om_(om),
      filePtr_(TFile::Open(file_.c_str(), "update", "", om_->compressionLevel())),
      fid_(),
      metaDataTree_(0),
      eventAux_(),
      lumiAux_(),
      runAux_(),
      pEventAux_(&eventAux_),
      pLumiAux_(&lumiAux_),
      pRunAux_(&runAux_),
      eventTree_(filePtr_, InEvent, pEventAux_, om_->basketSize(), om_->splitLevel(),
		  chains_.event_.get(), chains_.eventMeta_.get(), om_->droppedPriorVec()[InEvent]),
      lumiTree_(filePtr_, InLumi, pLumiAux_, om_->basketSize(), om_->splitLevel(),
		 chains_.lumi_.get(), chains_.lumiMeta_.get(), om_->droppedPriorVec()[InLumi]),
      runTree_(filePtr_, InRun, pRunAux_, om_->basketSize(), om_->splitLevel(),
		chains_.run_.get(), chains_.runMeta_.get(), om_->droppedPriorVec()[InRun]),
      treePointers_(),
      provenances_(),
      newFileAtEndOfRun_(false) {
    treePointers_[InEvent] = &eventTree_;
    treePointers_[InLumi]  = &lumiTree_;
    treePointers_[InRun]   = &runTree_;
    TTree::SetMaxTreeSize(kMaxLong64);

    for (int i = InEvent; i < EndBranchType; ++i) {
      BranchType branchType = static_cast<BranchType>(i);
      OutputItemList & outputItemList = outputItemList_[branchType];
      Selections const& descVector = om_->descVec()[branchType];
      Selections const& droppedVector = om_->droppedVec()[branchType];
      
      for (Selections::const_iterator it = descVector.begin(), itEnd = descVector.end(); it != itEnd; ++it) {
        BranchDescription const& prod = **it;
        outputItemList.push_back(OutputItem(&prod, true));
      }
      for (Selections::const_iterator it = droppedVector.begin(), itEnd = droppedVector.end(); it != itEnd; ++it) {
        BranchDescription const& prod = **it;
        outputItemList.push_back(OutputItem(&prod, false));
      }
      for (OutputItemList::iterator it = outputItemList.begin(), itEnd = outputItemList.end(); it != itEnd; ++it) {
	treePointers_[branchType]->addBranch(*it->branchDescription_, it->selected_, it->branchEntryDescription_, it->product_);
      }
    }

    // Don't split metadata tree.
    metaDataTree_ = RootOutputTree::makeTree(filePtr_.get(), poolNames::metaDataTreeName(), 0, 0);

    pool::Guid guid;
    pool::Guid::create(guid);

    fid_ = guid.toString();

    // Register the output file with the JobReport service
    // and get back the token for it.
    std::string moduleName = "PoolOutputModule";
    Service<JobReport> reportSvc;
    reportToken_ = reportSvc->outputFileOpened(
		      file_, logicalFile_,  // PFN and LFN
		      om_->catalog_,  // catalog
		      moduleName,   // module class name
		      om_->moduleLabel_,  // module label
		      fid_, // file id (guid)
		      eventTree_.branchNames()); // branch names being written
  }

  void RootOutputFile::writeOne(EventPrincipal const& e) {
    ++eventCount_;
    // Write auxiliary branch

    pEventAux_ = &e.aux();

    fillBranches(InEvent, e.groupGetter());

    // Report event written 
    Service<JobReport> reportSvc;
    reportSvc->eventWrittenToFile(reportToken_, e.id().run(), e.id().event());

    if (eventCount_ >= fileSizeCheckEvent_) {
	unsigned int const oneK = 1024;
	Long64_t size = filePtr_->GetSize()/oneK;
	unsigned int eventSize = std::max(size/eventCount_, 1LL);
	if (size + 2*eventSize >= om_->maxFileSize_) {
	  newFileAtEndOfRun_ = true;
	} else {
	  unsigned int increment = (om_->maxFileSize_ - size)/eventSize;
	  increment -= increment/8;	// Prevents overshoot
	  fileSizeCheckEvent_ = eventCount_ + increment;
	}
    }
  }

  void RootOutputFile::writeLuminosityBlock(LuminosityBlockPrincipal const& lb) {
    // Write auxiliary branch
    pLumiAux_ = &lb.aux();
    fillBranches(InLumi, lb.groupGetter());
  }

  bool RootOutputFile::writeRun(RunPrincipal const& r) {
    // Write auxiliary branch
    pRunAux_ = &r.aux();
    fillBranches(InRun, r.groupGetter());
    return newFileAtEndOfRun_;
  }

  void RootOutputFile::writeFileFormatVersion() {
    FileFormatVersion fileFormatVersion(edm::getFileFormatVersion());
    FileFormatVersion * pFileFmtVsn = &fileFormatVersion;
    TBranch* b = metaDataTree_->Branch(poolNames::fileFormatVersionBranchName().c_str(), &pFileFmtVsn, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeProcessConfigurationRegistry() {
    // We don't do this yet; currently we're storing a slightly bloated ProcessHistoryRegistry.
  }

  void RootOutputFile::writeProcessHistoryRegistry() { 
    ProcessHistoryMap *pProcHistMap = &ProcessHistoryRegistry::instance()->data();
    TBranch* b = metaDataTree_->Branch(poolNames::processHistoryMapBranchName().c_str(), &pProcHistMap, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeModuleDescriptionRegistry() { 
    ModuleDescriptionMap *pModDescMap = &ModuleDescriptionRegistry::instance()->data();
    TBranch* b = metaDataTree_->Branch(poolNames::moduleDescriptionMapBranchName().c_str(), &pModDescMap, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeParameterSetRegistry() { 
    typedef std::map<ParameterSetID, ParameterSetBlob> ParameterSetMap;
    ParameterSetMap psetMap;
    pset::Registry const* psetRegistry = pset::Registry::instance();    
    for (pset::Registry::const_iterator it = psetRegistry->begin(), itEnd = psetRegistry->end(); it != itEnd; ++it) {
      psetMap.insert(std::make_pair(it->first, ParameterSetBlob(it->second.toStringOfTracked())));
    }
    ParameterSetMap *pPsetMap = &psetMap;
    TBranch* b = metaDataTree_->Branch(poolNames::parameterSetMapBranchName().c_str(), &pPsetMap, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeProductDescriptionRegistry() { 
    Service<ConstProductRegistry> reg;
    ProductRegistry pReg = reg->productRegistry();
    ProductRegistry * ppReg = &pReg;
    TBranch* b = metaDataTree_->Branch(poolNames::productDescriptionBranchName().c_str(), &ppReg, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::finishEndFile() { 
    metaDataTree_->SetEntries(-1);
    RootOutputTree::writeTTree(metaDataTree_);
    for (int i = InEvent; i < EndBranchType; ++i) {
      BranchType branchType = static_cast<BranchType>(i);
      buildIndex(treePointers_[branchType]->tree(), branchType);
      setBranchAliases(treePointers_[branchType]->tree(), om_->descVec()[branchType]);
      treePointers_[branchType]->writeTree();
    }

    // close the file -- mfp
    filePtr_->Close();
    filePtr_.reset();

    // report that file has been closed
    Service<JobReport> reportSvc;
    reportSvc->outputFileClosed(reportToken_);

  }

  void RootOutputFile::RootOutputFile::fillBranches(BranchType const& branchType, Principal const& principal) const {

    OutputItemList const& items = outputItemList_[branchType];
    // Loop over EDProduct branches, fill the provenance, and write the branch.
    for (OutputItemList::const_iterator i = items.begin(), iEnd = items.end();
	 i != iEnd; ++i) {
      ProductID const& id = i->branchDescription_->productID_;

      if (id == ProductID()) {
	throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	  << "PoolOutputModule::write: invalid ProductID supplied in productRegistry\n";
      }

      EDProduct const* product = 0;
      BasicHandle const bh = principal.getForOutput(id, i->selected_);
      if (bh.provenance() == 0) {
	// No group with this ID is in the event.
	// Create and write the provenance.
	if (i->branchDescription_->produced_) {
          BranchEntryDescription provenance;
	  provenance.moduleDescriptionID_ = i->branchDescription_->moduleDescriptionID_;
	  provenance.productID_ = id;
	  provenance.status_ = BranchEntryDescription::CreatorNotRun;
	  provenance.isPresent_ = false;
	  provenance.cid_ = 0;
	  
	  provenances_.push_front(provenance); 
	  i->branchEntryDescription_ = &*provenances_.begin();
	} else {
	    throw edm::Exception(errors::ProductNotFound,"NoMatch")
	      << "PoolOutputModule: Unexpected internal error.  Contact the framework group.\n"
	      << "No group for branch" << i->branchDescription_->branchName_ << '\n';
	}
      } else {
	product = bh.wrapper();
        BranchEntryDescription const& provenance = bh.provenance()->event();
	// There is a group with this ID is in the event.  Write the provenance.
	bool present = i->selected_ && product && product->isPresent();
	if (present == provenance.isPresent()) {
	  // The provenance can be written out as is, saving a copy. 
	  i->branchEntryDescription_ = &provenance;
	} else {
	  // We need to make a private copy of the provenance so we can set isPresent_ correctly.
	  provenances_.push_front(provenance);
	  provenances_.begin()->isPresent_ = present;
	  i->branchEntryDescription_ = &*provenances_.begin();
	}
      }
      if (i->selected_) {
	if (product == 0) {
	  // Add a null product.
          ROOT::Reflex::Object object = i->branchDescription_->type_.Construct();
    	  product = static_cast<EDProduct *>(object.Address());
	}
	i->product_ = product;
      }
    }
    treePointers_[branchType]->fillTree();
  }



  void
  RootOutputFile::buildIndex(TTree * tree, BranchType const& branchType) {

    if (tree->GetEntries() == 0) return;

    // BuildIndex must read the auxiliary branch, so the
    // buffers need to be set to point to allocated memory.
    pEventAux_ = &eventAux_;
    pLumiAux_ = &lumiAux_;
    pRunAux_ = &runAux_;

    std::string const aux = BranchTypeToAuxiliaryBranchName(branchType);
    std::string const majorID = aux + ".id_.run_";
    std::string const minorID = (branchType == InEvent ? aux + ".id_.event_" :
				(branchType == InLumi ? aux + ".id_.luminosityBlock_" : std::string()));
    
    if (minorID.empty()) {
      tree->BuildIndex(majorID.c_str());
    } else {
      tree->BuildIndex(majorID.c_str(), minorID.c_str());
    }
  }
  
  void
  RootOutputFile::setBranchAliases(TTree *tree, Selections const& branches) const {
    if (tree) {
      for (Selections::const_iterator i = branches.begin(), iEnd = branches.end();
	  i != iEnd; ++i) {
	BranchDescription const& pd = **i;
	std::string const& full = pd.branchName() + "obj";
	if (pd.branchAliases().empty()) {
	  std::string const& alias =
	      (pd.productInstanceName().empty() ? pd.moduleLabel() : pd.productInstanceName());
	  tree->SetAlias(alias.c_str(), full.c_str());
	} else {
	  std::set<std::string>::const_iterator it = pd.branchAliases().begin(), itEnd = pd.branchAliases().end();
	  for (; it != itEnd; ++it) {
	    tree->SetAlias((*it).c_str(), full.c_str());
	  }
	}
      }
    }
  }
}
