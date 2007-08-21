// $Id: RootOutputFile.cc,v 1.1 2007/08/20 23:45:05 wmtan Exp $

#include "IOPool/Output/src/PoolOutputModule.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h" 
#include "IOPool/Output/src/RootOutputFile.h"

#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "FWCore/Utilities/interface/GetFileFormatVersion.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Catalog/interface/FileCatalog.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TTree.h"
#include "TFile.h"
#include "Rtypes.h"

#include <map>
#include <iomanip>

namespace edm {
  RootOutputFile::RootOutputFile(PoolOutputModule *om, std::string const& fileName, std::string const& logicalFileName) :
      outputItemList_(), 
      file_(fileName),
      logicalFile_(logicalFileName),
      reportToken_(0),
      eventCount_(0),
      fileSizeCheckEvent_(100),
      om_(om),
      filePtr_(TFile::Open(file_.c_str(), "update", "", om_->compressionLevel())),
      metaDataTree_(0),
      eventAux_(),
      lumiAux_(),
      runAux_(),
      pEventAux_(&eventAux_),
      pLumiAux_(&lumiAux_),
      pRunAux_(&runAux_),
      eventTree_(InEvent, pEventAux_, om_->basketSize(), om_->splitLevel()),
      lumiTree_(InLumi, pLumiAux_, om_->basketSize(), om_->splitLevel()),
      runTree_(InRun, pRunAux_, om_->basketSize(), om_->splitLevel()),
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

    metaDataTree_ = new TTree(poolNames::metaDataTreeName().c_str(), "", 0);  // Don't split metadata tree.

    pool::FileCatalog::FileID fid = om_->catalog_.registerFile(file_, logicalFile_);

    // Register the output file with the JobReport service
    // and get back the token for it.
    std::string moduleName = "PoolOutputModule";
    Service<JobReport> reportSvc;
    reportToken_ = reportSvc->outputFileOpened(
		      file_, logicalFile_,  // PFN and LFN
		      om_->catalog_.url(),  // catalog
		      moduleName,   // module class name
		      om_->moduleLabel_,  // module label
		      fid, // file id (guid)
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
    if (eventCount_ % om_->commitInterval_ == 0) {
	// QQQ
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

  void RootOutputFile::fillBranches(BranchType const& branchType, Principal const& dataBlock) const {

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
      BasicHandle const bh = dataBlock.getForOutput(id, i->selected_);
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
	  std::string const& name = i->branchDescription_->className();
	  std::string const className = wrappedClassName(name);
	  TClass *cp = gROOT->GetClass(className.c_str());
	  if (cp == 0) {
	    throw edm::Exception(errors::ProductNotFound,"NoMatch")
	      << "TypeID::className: No dictionary for class " << className << '\n'
	      << "Add an entry for this class\n"
	      << "to the appropriate 'classes_def.xml' and 'classes.h' files." << '\n';
	  }
	  product = static_cast<EDProduct *>(cp->New());
	}
	i->product_ = product;
      }
    }
    treePointers_[branchType]->fillTree();
  }

  void RootOutputFile::endFile() {
    FileFormatVersion fileFormatVersion(edm::getFileFormatVersion());
    FileFormatVersion * pFileFmtVsn = &fileFormatVersion;
    metaDataTree_->Branch(poolNames::fileFormatVersionBranchName().c_str(), &pFileFmtVsn, om_->basketSize(), 0);

    ModuleDescriptionMap *pModDescMap = &ModuleDescriptionRegistry::instance()->data();
    metaDataTree_->Branch(poolNames::moduleDescriptionMapBranchName().c_str(), &pModDescMap, om_->basketSize(), 0);

    ProcessHistoryMap *pProcHistMap = &ProcessHistoryRegistry::instance()->data();
    metaDataTree_->Branch(poolNames::processHistoryMapBranchName().c_str(), &pProcHistMap, om_->basketSize(), 0);

    typedef std::map<ParameterSetID, ParameterSetBlob> ParameterSetMap;
    ParameterSetMap psetMap;
    pset::Registry const* psetRegistry = pset::Registry::instance();    
    for (pset::Registry::const_iterator it = psetRegistry->begin(), itEnd = psetRegistry->end(); it != itEnd; ++it) {
      psetMap.insert(std::make_pair(it->first, ParameterSetBlob(it->second.toStringOfTracked())));
    }
    ParameterSetMap *pPsetMap = &psetMap;
    metaDataTree_->Branch(poolNames::parameterSetMapBranchName().c_str(), &pPsetMap, om_->basketSize(), 0);

    Service<ConstProductRegistry> reg;
    ProductRegistry pReg = reg->productRegistry();
    ProductRegistry * ppReg = &pReg;
    metaDataTree_->Branch(poolNames::productDescriptionBranchName().c_str(), &ppReg, om_->basketSize(), 0);

    metaDataTree_->Fill();

    rootPostProcess();

    metaDataTree_->Write();
    for (int i = InEvent; i < EndBranchType; ++i) {
      BranchType branchType = static_cast<BranchType>(i);
      treePointers_[branchType]->writeTree();
    }

    om_->catalog_.commitCatalog();
    // report that file has been closed
    Service<JobReport> reportSvc;
    reportSvc->outputFileClosed(reportToken_);
  }


  void
  RootOutputFile::rootPostProcess() {
    TTree *tEvent = eventTree_.tree();
    TTree *tLumi = lumiTree_.tree();
    TTree *tRun = runTree_.tree();

    setBranchAliases(tEvent, om_->descVec()[InEvent]);
    setBranchAliases(tLumi, om_->descVec()[InLumi]);
    setBranchAliases(tRun, om_->descVec()[InRun]);

    // BuildIndex must read the auxiliary branch, so the
    // buffers need to be set to point to allocated memory.
    pEventAux_ = &eventAux_;
    pLumiAux_ = &lumiAux_;
    pRunAux_ = &runAux_;

    tEvent->BuildIndex("id_.run_", "id_.event_");
    tLumi->BuildIndex("id_.run_", "id_.luminosityBlock_");
    tRun->BuildIndex("id_.run_");
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
