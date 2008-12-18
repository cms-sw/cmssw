
#include "IOPool/Output/src/RootOutputFile.h"

#include "FWCore/Utilities/interface/GlobalIdentifier.h"

#include "DataFormats/Provenance/interface/EventAuxiliary.h" 
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h" 
#include "DataFormats/Provenance/interface/RunAuxiliary.h" 
#include "FWCore/Version/interface/GetFileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/Provenance/interface/BranchChildren.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Provenance/interface/BranchIDListRegistry.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TROOT.h"

#include "TTree.h"
#include "TFile.h"
#include "TClass.h"
#include "Rtypes.h"

#include <algorithm>
#include <iomanip>
#include <sstream>


namespace edm {

  namespace {
    bool
    sorterForJobReportHash(BranchDescription const* lh, BranchDescription const* rh) {
      return 
	lh->fullClassName() < rh->fullClassName() ? true :
	lh->fullClassName() > rh->fullClassName() ? false :
	lh->moduleLabel() < rh->moduleLabel() ? true :
	lh->moduleLabel() > rh->moduleLabel() ? false :
	lh->productInstanceName() < rh->productInstanceName() ? true :
	lh->productInstanceName() > rh->productInstanceName() ? false :
	lh->processName() < rh->processName() ? true :
	false;
    }
  }

  RootOutputFile::RootOutputFile(PoolOutputModule *om, std::string const& fileName, std::string const& logicalFileName) :
      file_(fileName),
      logicalFile_(logicalFileName),
      reportToken_(0),
      om_(om),
      currentlyFastCloning_(),
      filePtr_(TFile::Open(file_.c_str(), "recreate", "", om_->compressionLevel())),
      fid_(),
      fileIndex_(),
      eventEntryNumber_(0LL),
      lumiEntryNumber_(0LL),
      runEntryNumber_(0LL),
      metaDataTree_(0),
      parentageTree_(0),
      eventHistoryTree_(0),
      pEventAux_(0),
      pLumiAux_(0),
      pRunAux_(0),
      eventEntryInfoVector_(),
      lumiEntryInfoVector_(),
      runEntryInfoVector_(),
      pEventEntryInfoVector_(&eventEntryInfoVector_),
      pLumiEntryInfoVector_(&lumiEntryInfoVector_),
      pRunEntryInfoVector_(&runEntryInfoVector_),
      pHistory_(0),
      eventTree_(static_cast<EventPrincipal *>(0),
                 filePtr_, InEvent, pEventAux_, pEventEntryInfoVector_,
                 om_->basketSize(), om_->splitLevel(), om_->treeMaxVirtualSize()),
      lumiTree_(static_cast<LuminosityBlockPrincipal *>(0),
                filePtr_, InLumi, pLumiAux_, pLumiEntryInfoVector_,
                om_->basketSize(), om_->splitLevel(), om_->treeMaxVirtualSize()),
      runTree_(static_cast<RunPrincipal *>(0),
               filePtr_, InRun, pRunAux_, pRunEntryInfoVector_,
               om_->basketSize(), om_->splitLevel(), om_->treeMaxVirtualSize()),
      treePointers_(),
      dataTypeReported_(false) {
    treePointers_[InEvent] = &eventTree_;
    treePointers_[InLumi]  = &lumiTree_;
    treePointers_[InRun]   = &runTree_;

    for (int i = InEvent; i < NumBranchTypes; ++i) {
      BranchType branchType = static_cast<BranchType>(i);
      for (OutputItemList::const_iterator it = om_->selectedOutputItemList()[branchType].begin(),
	  itEnd = om_->selectedOutputItemList()[branchType].end();
	  it != itEnd; ++it) {
	treePointers_[branchType]->addBranch(*it->branchDescription_,
					      it->product_,
					      it->branchDescription_->produced());
      }
    }
    // Don't split metadata tree or event description tree
    metaDataTree_         = RootOutputTree::makeTTree(filePtr_.get(), poolNames::metaDataTreeName(), 0);
    parentageTree_ = RootOutputTree::makeTTree(filePtr_.get(), poolNames::parentageTreeName(), 0);

    // Create the tree that will carry (event) History objects.
    eventHistoryTree_     = RootOutputTree::makeTTree(filePtr_.get(), poolNames::eventHistoryTreeName(), om_->splitLevel());
    if (!eventHistoryTree_)
      throw edm::Exception(edm::errors::FatalRootError) 
	<< "Failed to create the tree for History objects\n";

    if (! eventHistoryTree_->Branch(poolNames::eventHistoryBranchName().c_str(), &pHistory_, om_->basketSize(), 0))
      throw edm::Exception(edm::errors::FatalRootError) 
	<< "Failed to create a branch for Historys in the output file\n";

    fid_ = FileID(createGlobalIdentifier());

    // For the Job Report, get a vector of branch names in the "Events" tree.
    // Also create a hash of all the branch names in the "Events" tree
    // in a deterministic order, except use the full class name instead of the friendly class name.
    // To avoid extra string copies, we create a vector of pointers into the product registry,
    // and use a custom comparison operator for sorting.
    std::vector<std::string> branchNames;
    std::vector<BranchDescription const*> branches;
    branchNames.reserve(om_->selectedOutputItemList()[InEvent].size());
    branches.reserve(om->selectedOutputItemList()[InEvent].size());
    for (OutputItemList::const_iterator it = om_->selectedOutputItemList()[InEvent].begin(),
	  itEnd = om_->selectedOutputItemList()[InEvent].end();
	  it != itEnd; ++it) {
      branchNames.push_back(it->branchDescription_->branchName());
      branches.push_back(it->branchDescription_);
    }
    // Now sort the branches for the hash.
    sort_all(branches, sorterForJobReportHash);
    // Now, make a concatenated string.
    std::ostringstream oss;
    char const underscore = '_';
    for (std::vector<BranchDescription const*>::const_iterator it = branches.begin(), itEnd = branches.end(); it != itEnd; ++it) {
      BranchDescription const& bd = **it;
      oss <<  bd.fullClassName() << underscore
	  << bd.moduleLabel() << underscore
	  << bd.productInstanceName() << underscore
	  << bd.processName() << underscore;
    }
    std::string stringrep = oss.str();
    cms::Digest md5alg(stringrep);

    // Register the output file with the JobReport service
    // and get back the token for it.
    std::string moduleName = "PoolOutputModule";
    Service<JobReport> reportSvc;
    reportToken_ = reportSvc->outputFileOpened(
		      file_, logicalFile_,  // PFN and LFN
		      om_->catalog_,  // catalog
		      moduleName,   // module class name
		      om_->moduleLabel_,  // module label
		      fid_.fid(), // file id (guid)
		      std::string(), // data type (not yet known, so string is empty).
		      md5alg.digest().toString(), // branch hash
		      branchNames); // branch names being written
  }

  void RootOutputFile::beginInputFile(FileBlock const& fb, bool fastClone) {

    currentlyFastCloning_ = om_->fastCloning() && fb.fastClonable() && fastClone;
    if (currentlyFastCloning_) currentlyFastCloning_ = eventTree_.checkSplitLevelAndBasketSize(fb.tree());

    eventTree_.beginInputFile(currentlyFastCloning_);
    eventTree_.fastCloneTree(fb.tree());
  }

  void RootOutputFile::respondToCloseInputFile(FileBlock const&) {
    eventTree_.setEntries();
    lumiTree_.setEntries();
    runTree_.setEntries();
  }

  bool RootOutputFile::shouldWeCloseFile() const {
    unsigned int const oneK = 1024;
    Long64_t size = filePtr_->GetSize()/oneK;
    return(size >= om_->maxFileSize_);
  }

  void RootOutputFile::writeOne(EventPrincipal const& e) {
    // Auxiliary branch
    pEventAux_ = &e.aux();
   
    // Store an invailid process history ID in EventAuxiliary for obsolete field.
    pEventAux_->processHistoryID_ = ProcessHistoryID(); // backward compatibility
    
    // Because getting the data may cause an exception to be thrown we want to do that
    // first before writing anything to the file about this event
    // NOTE: pEventAux_ must be set before calling fillBranches since it gets written out
    // in that routine.
    fillBranches(InEvent, e, pEventEntryInfoVector_);
     
    // History branch
    History historyForOutput(e.history());
    historyForOutput.addEventSelectionEntry(om_->selectorConfig());
    pHistory_ = &historyForOutput;
    int sz = eventHistoryTree_->Fill();
    if ( sz <= 0)
      throw edm::Exception(edm::errors::FatalRootError) 
	<< "Failed to fill the History tree for event: " << e.id()
	<< "\nTTree::Fill() returned " << sz << " bytes written." << std::endl;

    // Add the dataType to the job report if it hasn't already been done
    if(!dataTypeReported_) {
      Service<JobReport> reportSvc;
      std::string dataType("MC");
      if(pEventAux_->isRealData())  dataType = "Data";
      reportSvc->reportDataType(reportToken_,dataType);
      dataTypeReported_ = true;
    }

    pHistory_ = & e.history();

    // Add event to index
    fileIndex_.addEntry(pEventAux_->run(), pEventAux_->luminosityBlock(), pEventAux_->event(), eventEntryNumber_);
    ++eventEntryNumber_;

    // Report event written 
    Service<JobReport> reportSvc;
    reportSvc->eventWrittenToFile(reportToken_, e.id().run(), e.id().event());
  }

  void RootOutputFile::writeLuminosityBlock(LuminosityBlockPrincipal const& lb) {
    // Auxiliary branch
    pLumiAux_ = &lb.aux();
    // Add lumi to index.
    fileIndex_.addEntry(pLumiAux_->run(), pLumiAux_->luminosityBlock(), 0U, lumiEntryNumber_);
    ++lumiEntryNumber_;
    fillBranches(InLumi, lb, pLumiEntryInfoVector_);
  }

  void RootOutputFile::writeRun(RunPrincipal const& r) {
    // Auxiliary branch
    pRunAux_ = &r.aux();
    // Add run to index.
    fileIndex_.addEntry(pRunAux_->run(), 0U, 0U, runEntryNumber_);
    ++runEntryNumber_;
    fillBranches(InRun, r, pRunEntryInfoVector_);
  }

  void RootOutputFile::writeParentageRegistry() {
    ParentageID const* hash(0);
    Parentage const*   desc(0);
    
    if (!parentageTree_->Branch(poolNames::parentageIDBranchName().c_str(), 
					&hash, om_->basketSize(), 0))
      throw edm::Exception(edm::errors::FatalRootError) 
	<< "Failed to create a branch for ParentageIDs in the output file";

    if (!parentageTree_->Branch(poolNames::parentageBranchName().c_str(), 
					&desc, om_->basketSize(), 0))
      throw edm::Exception(edm::errors::FatalRootError) 
	<< "Failed to create a branch for Parentages in the output file";

    ParentageRegistry& ptReg = *ParentageRegistry::instance();
    for (ParentageRegistry::const_iterator
	   i = ptReg.begin(),
	   e = ptReg.end();
	 i != e;
	 ++i) {
	hash = const_cast<ParentageID*>(&(i->first)); // cast needed because keys are const
	desc = &(i->second);
	parentageTree_->Fill();
      }
  }

  void RootOutputFile::writeFileFormatVersion() {
    FileFormatVersion fileFormatVersion(getFileFormatVersion());
    FileFormatVersion * pFileFmtVsn = &fileFormatVersion;
    TBranch* b = metaDataTree_->Branch(poolNames::fileFormatVersionBranchName().c_str(), &pFileFmtVsn, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeFileIdentifier() {
    FileID *fidPtr = &fid_;
    TBranch* b = metaDataTree_->Branch(poolNames::fileIdentifierBranchName().c_str(), &fidPtr, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeFileIndex() {
    fileIndex_.sortBy_Run_Lumi_Event();
    FileIndex *findexPtr = &fileIndex_;
    TBranch* b = metaDataTree_->Branch(poolNames::fileIndexBranchName().c_str(), &findexPtr, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeEventHistory() {
    RootOutputTree::writeTTree(eventHistoryTree_);
  }

  void RootOutputFile::writeProcessConfigurationRegistry() {
/*
    ProcessConfigurationRegistry::collection_type *p = &ProcessConfigurationRegistry::instance()->data();
    TBranch* b = metaDataTree_->Branch(poolNames::processConfigurationBranchName().c_str(), &p, om_->basketSize(), 0);
    assert(b);
    b->Fill();
*/
  }

  void RootOutputFile::writeProcessHistoryRegistry() { 
    ProcessHistoryRegistry::collection_type *p = &ProcessHistoryRegistry::instance()->data();
    TBranch* b = metaDataTree_->Branch(poolNames::processHistoryMapBranchName().c_str(), &p, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeBranchIDListRegistry() { 
    BranchIDListRegistry::collection_type *p = &BranchIDListRegistry::instance()->data();
    TBranch* b = metaDataTree_->Branch(poolNames::branchIDListBranchName().c_str(), &p, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeParameterSetRegistry() { 
    typedef std::map<ParameterSetID, ParameterSetBlob> ParameterSetMap;
    ParameterSetMap psetMap;
    pset::fillMap(pset::Registry::instance(), psetMap);
    ParameterSetMap *pPsetMap = &psetMap;
    TBranch* b = metaDataTree_->Branch(poolNames::parameterSetMapBranchName().c_str(), &pPsetMap, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeProductDescriptionRegistry() { 
    // Make a local copy of the ProductRegistry, removing any transient or pruned products.
    typedef ProductRegistry::ProductList ProductList;
    edm::Service<edm::ConstProductRegistry> reg;
    ProductRegistry pReg(reg->productList());
    ProductList & pList  = const_cast<ProductList &>(pReg.productList());
    std::set<BranchID>::iterator end = branchesWithStoredHistory_.end();
    for (ProductList::iterator it = pList.begin(); it != pList.end(); ) {
      if (branchesWithStoredHistory_.find(it->second.branchID()) == end) {
	// avoid invalidating iterator on deletion
	ProductList::iterator itCopy = it;
	++it;
	pList.erase(itCopy);
	
      } else {
	++it;
      }
    }

    ProductRegistry * ppReg = &pReg;
    TBranch* b = metaDataTree_->Branch(poolNames::productDescriptionBranchName().c_str(), &ppReg, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  } 
  void RootOutputFile::writeProductDependencies() { 
    BranchChildren& pDeps = const_cast<BranchChildren&>(om_->branchChildren());
    BranchChildren * ppDeps = &pDeps;
    TBranch* b = metaDataTree_->Branch(poolNames::productDependenciesBranchName().c_str(), &ppDeps, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::finishEndFile() { 
    metaDataTree_->SetEntries(-1);
    RootOutputTree::writeTTree(metaDataTree_);

    RootOutputTree::writeTTree(parentageTree_);

    // Create branch aliases for all the branches in the
    // events/lumis/runs trees. The loop is over all types of data
    // products.
    for (int i = InEvent; i < NumBranchTypes; ++i) {
      BranchType branchType = static_cast<BranchType>(i);
      setBranchAliases(treePointers_[branchType]->tree(), om_->keptProducts()[branchType]);
      treePointers_[branchType]->writeTree();
    }

    // close the file -- mfp
    filePtr_->Close();
    filePtr_.reset();

    // report that file has been closed
    Service<JobReport> reportSvc;
    reportSvc->outputFileClosed(reportToken_);

  }

  void
  RootOutputFile::setBranchAliases(TTree *tree, Selections const& branches) const {
    if (tree && tree->GetNbranches() != 0) {
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
   
  void
  RootOutputFile::insertAncestors(ProductProvenance const& iGetParents,
                                  Principal const& principal,
                                  std::set<ProductProvenance>& oToFill) {
    if(om_->dropMetaData() == PoolOutputModule::DropAll) return;
    if(om_->dropMetaDataForDroppedData()) return;
    BranchMapper const& iMapper = *principal.branchMapperPtr();
    std::vector<BranchID> const& parentIDs = iGetParents.parentage().parents();
    for(std::vector<BranchID>::const_iterator it=parentIDs.begin(), itEnd = parentIDs.end();
          it != itEnd; ++it) {
      branchesWithStoredHistory_.insert(*it);
      boost::shared_ptr<ProductProvenance> info = iMapper.branchIDToProvenance(*it);
      if(info) {
        if(om_->dropMetaData() == PoolOutputModule::DropNone ||
		 principal.getProvenance(info->branchID()).product().produced()) {
	  if(oToFill.insert(*info).second) {
            //haven't seen this one yet
            insertAncestors(*info, principal, oToFill);
	  }
	}
      }
    }
  }
   
  void RootOutputFile::fillBranches(
		BranchType const& branchType,
		Principal const& principal,
		std::vector<ProductProvenance>* productProvenanceVecPtr) {

    std::vector<boost::shared_ptr<EDProduct> > dummies;

    bool const fastCloning = (branchType == InEvent) && currentlyFastCloning_;
    
    OutputItemList const& items = om_->selectedOutputItemList()[branchType];

    std::set<ProductProvenance> provenanceToKeep;

    // Loop over EDProduct branches, fill the provenance, and write the branch.
    for (OutputItemList::const_iterator i = items.begin(), iEnd = items.end(); i != iEnd; ++i) {

      BranchID const& id = i->branchDescription_->branchID();
      branchesWithStoredHistory_.insert(id);
       
      bool produced = i->branchDescription_->produced();
      bool keepProvenance = om_->dropMetaData() == PoolOutputModule::DropNone ||
			   (om_->dropMetaData() == PoolOutputModule::DropPrior && produced);
      bool getProd = (produced || !fastCloning ||
	 treePointers_[branchType]->uncloned(i->branchDescription_->branchName()));

      EDProduct const* product = 0;
      OutputHandle const oh = principal.getForOutput(id, getProd);
      if (!oh.productProvenance()) {
	// No product with this ID is in the event.
	// Create and write the provenance.
	if (keepProvenance) {
	  if (produced) {
            provenanceToKeep.insert(ProductProvenance(i->branchDescription_->branchID(),
		        productstatus::neverCreated()));
	  } else {
            provenanceToKeep.insert(ProductProvenance(i->branchDescription_->branchID(),
		        productstatus::dropped()));
	  }
	}
      } else {
	product = oh.wrapper();
	if (keepProvenance) {
	  provenanceToKeep.insert(*oh.productProvenance());
	  assert(principal.branchMapperPtr());
	  insertAncestors(*oh.productProvenance(), principal, provenanceToKeep);
	}
      }
      if (getProd) {
	if (product == 0) {
	  // No product with this ID is in the event.
	  // Add a null product.
	  TClass *cp = gROOT->GetClass(i->branchDescription_->wrappedName().c_str());
	  boost::shared_ptr<EDProduct> dummy(static_cast<EDProduct *>(cp->New()));
	  dummies.push_back(dummy);
	  product = dummy.get();
	}
	i->product_ = product;
      }
    }
     
    productProvenanceVecPtr->assign(provenanceToKeep.begin(), provenanceToKeep.end());
    treePointers_[branchType]->fillTree();
    productProvenanceVecPtr->clear();
  }

}
