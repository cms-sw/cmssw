// $Id: RootOutputFile.cc,v 1.51 2008/03/05 05:55:31 wmtan Exp $

#include "RootOutputFile.h"
#include "PoolOutputModule.h"

#include "FWCore/Utilities/interface/GlobalIdentifier.h"

#include "DataFormats/Provenance/interface/EventAuxiliary.h" 
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "FWCore/Utilities/interface/GetFileFormatVersion.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/Provenance/interface/EntryDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TROOT.h"
#include "TBranchElement.h"
#include "TObjArray.h"
#include "TTree.h"
#include "TFile.h"
#include "TClass.h"
#include "Rtypes.h"

#include <algorithm>
#include <map>
#include <iomanip>

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

  RootOutputFile::OutputItem::Sorter::Sorter(TTree * tree) {
    // Fill a map mapping branch names to an index specifying the order in the tree.
    if (tree != 0) {
      TObjArray * branches = tree->GetListOfBranches();
      for (int i = 0; i < branches->GetEntries(); ++i) {
        TBranchElement * br = (TBranchElement *)branches->At(i);
        treeMap_.insert(std::make_pair(std::string(br->GetName()), i));
      }
    }
  }

  bool
  RootOutputFile::OutputItem::Sorter::operator()(OutputItem const& lh, OutputItem const& rh) const {
    // Provides a comparison for sorting branches according to the index values in treeMap_.
    // Branches not found are always put at the end (i.e. not found > found).
    if (treeMap_.empty()) return lh < rh;
    std::string const& lstring = lh.branchDescription_->branchName();
    std::string const& rstring = rh.branchDescription_->branchName();
    std::map<std::string, int>::const_iterator lit = treeMap_.find(lstring);
    std::map<std::string, int>::const_iterator rit = treeMap_.find(rstring);
    bool lfound = (lit != treeMap_.end());
    bool rfound = (rit != treeMap_.end());
    if (lfound && rfound) {
      return lit->second < rit->second;
    } else if (lfound) { 
      return true;
    } else if (rfound) { 
      return false;
    }
    return lh < rh;
  }

  RootOutputFile::RootOutputFile(PoolOutputModule *om, std::string const& fileName, std::string const& logicalFileName) :
      outputItemList_(), 
      file_(fileName),
      logicalFile_(logicalFileName),
      reportToken_(0),
      eventCount_(0),
      fileSizeCheckEvent_(100),
      om_(om),
      currentlyFastCloning_(),
      currentlyFastMetaCloning_(),
      filePtr_(TFile::Open(file_.c_str(), "recreate", "", om_->compressionLevel())),
      fid_(),
      fileIndex_(),
      eventEntryNumber_(0LL),
      lumiEntryNumber_(0LL),
      runEntryNumber_(0LL),
      metaDataTree_(0),
      entryDescriptionTree_(0),
      eventHistoryTree_(0),
      pEventAux_(0),
      pLumiAux_(0),
      pRunAux_(0),
      pProductStatuses_(0),
      pHistory_(0),
      eventTree_(filePtr_, InEvent, pEventAux_, pProductStatuses_, om_->basketSize(), om_->splitLevel()),
      lumiTree_(filePtr_, InLumi, pLumiAux_, pProductStatuses_, om_->basketSize(), om_->splitLevel()),
      runTree_(filePtr_, InRun, pRunAux_, pProductStatuses_, om_->basketSize(), om_->splitLevel()),
      treePointers_(),
      newFileAtEndOfRun_(false), 
      dataTypeReported_(false)  {
    treePointers_[InEvent] = &eventTree_;
    treePointers_[InLumi]  = &lumiTree_;
    treePointers_[InRun]   = &runTree_;
    TTree::SetMaxTreeSize(kMaxLong64);

    for (int i = InEvent; i < NumBranchTypes; ++i) {
      BranchType branchType = static_cast<BranchType>(i);
      TTree * meta = (branchType == InEvent ? om_->fileBlock_->metaTree() : 
		     (branchType == InLumi ? om_->fileBlock_->lumiMetaTree() : om_->fileBlock_->runMetaTree()));
      fillItemList(om_->keptProducts()[branchType], om_->droppedProducts()[branchType], outputItemList_[branchType], meta);
      int nBranches = (meta ? meta->GetNbranches() : 0);
      // itMarker Marks one past the end of the branches thast are in the input file.
      // needed for fast cloning. 
      OutputItemList::const_iterator itMarker = outputItemList_[branchType].begin() + nBranches;
      for (OutputItemList::const_iterator it = outputItemList_[branchType].begin(),
	  itEnd = outputItemList_[branchType].end();
	  it != itEnd; ++it) {
	treePointers_[branchType]->addBranch(*it->branchDescription_,
					      it->selected_,
					      it->entryDescriptionIDPtr_,
					      it->product_,
					      it < itMarker);
      }
    }
    // Don't split metadata tree or event description tree
    metaDataTree_         = RootOutputTree::makeTTree(filePtr_.get(), poolNames::metaDataTreeName(), 0);
    entryDescriptionTree_ = RootOutputTree::makeTTree(filePtr_.get(), poolNames::entryDescriptionTreeName(), 0);

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
    branchNames.reserve(outputItemList_[InEvent].size());
    branches.reserve(outputItemList_[InEvent].size());
    for (OutputItemList::const_iterator it = outputItemList_[InEvent].begin(),
	  itEnd = outputItemList_[InEvent].end();
	  it != itEnd; ++it) {
      if (it->selected_) {
	branchNames.push_back(it->branchDescription_->branchName());
	branches.push_back(it->branchDescription_);
      }
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

  void RootOutputFile::fillItemList(Selections const& keptVector,
				    Selections const& droppedVector,
				    OutputItemList & outputItemList,
				    TTree * meta) {

    // Fill outputItemList with an entry for each branch, including dropped branches.
    std::vector<std::string> const& renamed = om_->fileBlock_->sortedNewBranchNames();
    for (Selections::const_iterator it = keptVector.begin(), itEnd = keptVector.end(); it != itEnd; ++it) {
      BranchDescription const& prod = **it;
      outputItemList.push_back(OutputItem(&prod, true, binary_search_all(renamed, prod.branchName())));
    }
    for (Selections::const_iterator it = droppedVector.begin(), itEnd = droppedVector.end(); it != itEnd; ++it) {
      BranchDescription const& prod = **it;
      outputItemList.push_back(OutputItem(&prod, false, binary_search_all(renamed, prod.branchName())));
    }
    // Sort outputItemList to allow fast copying.
    // meta is a pointer to the input XMetaData tree (X is Event, Run, or LuminosityBlock).
    // The branches in outputItemList must be in the same order as in the input tree, with all new branches at the end.
    sort_all(outputItemList, OutputItem::Sorter(meta));
  }


  void RootOutputFile::beginInputFile(FileBlock const& fb, bool fastClone, bool fastMetaClone) {
    currentlyFastCloning_ = om_->fastCloning() && fb.fastClonable() && fastClone;
    currentlyFastMetaCloning_ = om_->fastCloning() && fb.fastClonable() && fastMetaClone;
    eventTree_.beginInputFile(currentlyFastCloning_, currentlyFastMetaCloning_);
    eventTree_.fastCloneTree(fb.tree(), fb.metaTree());
  }

  void RootOutputFile::respondToCloseInputFile(FileBlock const&) {
    eventTree_.setEntries();
    lumiTree_.setEntries();
    runTree_.setEntries();
  }

  void RootOutputFile::writeOne(EventPrincipal const& e) {
    ++eventCount_;

    // Auxiliary branch
    pEventAux_ = &e.aux();

    // History branch
    History historyForOutput(e.history());
    historyForOutput.addEntry(om_->selectorConfig());
    historyForOutput.setProcessHistoryID(pEventAux_->processHistoryID());
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

    // Product Statuses
    pProductStatuses_ = &e.productStatuses();
    pHistory_ = & e.history();

    // Add event to index
    fileIndex_.addEntry(pEventAux_->run(), pEventAux_->luminosityBlock(), pEventAux_->event(), eventEntryNumber_);
    ++eventEntryNumber_;

    // Store an invailid process history ID in EventAuxiliary for obsolete field.
    pEventAux_->processHistoryID_ = ProcessHistoryID();

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
    // Auxiliary branch
    pLumiAux_ = &lb.aux();
    // Product Statuses
    pProductStatuses_ = &lb.productStatuses();
    // Add lumi to index.
    fileIndex_.addEntry(pLumiAux_->run(), pLumiAux_->luminosityBlock(), 0U, lumiEntryNumber_);
    ++lumiEntryNumber_;
    fillBranches(InLumi, lb.groupGetter());
  }

  bool RootOutputFile::writeRun(RunPrincipal const& r) {
    // Auxiliary branch
    pRunAux_ = &r.aux();
    // Product Statuses
    pProductStatuses_ = &r.productStatuses();
    // Add run to index.
    fileIndex_.addEntry(pRunAux_->run(), 0U, 0U, runEntryNumber_);
    ++runEntryNumber_;
    fillBranches(InRun, r.groupGetter());
    return newFileAtEndOfRun_;
  }

  void RootOutputFile::writeEntryDescriptions() {
    EntryDescriptionID const* hash(0);
    EntryDescription const*   desc(0);
    
    if (! entryDescriptionTree_->Branch(poolNames::entryDescriptionIDBranchName().c_str(), 
					&hash, om_->basketSize(), 0))
      throw edm::Exception(edm::errors::FatalRootError) 
	<< "Failed to create a branch for EntryDescriptionIDs in the output file";

    if (! entryDescriptionTree_->Branch(poolNames::entryDescriptionBranchName().c_str(), 
					&desc, om_->basketSize(), 0))
      throw edm::Exception(edm::errors::FatalRootError) 
	<< "Failed to create a branch for EntryDescriptions in the output file";

    EntryDescriptionRegistry& edreg = *EntryDescriptionRegistry::instance();
    for (EntryDescriptionRegistry::const_iterator
	   i = edreg.begin(),
	   e = edreg.end();
	 i != e;
	 ++i)
      {
	hash = const_cast<EntryDescriptionID*>(&(i->first)); // cast needed because keys are const
	desc = &(i->second);
	entryDescriptionTree_->Fill();
      }
  }

  void RootOutputFile::writeFileFormatVersion() {
    FileFormatVersion fileFormatVersion(edm::getFileFormatVersion());
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
    fileIndex_.sort();
    FileIndex *findexPtr = &fileIndex_;
    TBranch* b = metaDataTree_->Branch(poolNames::fileIndexBranchName().c_str(), &findexPtr, om_->basketSize(), 0);
    assert(b);
    b->Fill();
  }

  void RootOutputFile::writeEventHistory() {
    RootOutputTree::writeTTree(eventHistoryTree_);
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

    //entryDescriptionTree_->SetEntries(-1);
    RootOutputTree::writeTTree(entryDescriptionTree_);

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

  void RootOutputFile::RootOutputFile::fillBranches(BranchType const& branchType, Principal const& principal) const {

    bool const fastCloning = (branchType == InEvent) && currentlyFastCloning_;
    bool const fastMetaCloning = (branchType == InEvent) && currentlyFastMetaCloning_;
    
    OutputItemList const& items = outputItemList_[branchType];

    // Loop over EDProduct branches, fill the provenance, and write the branch.
    for (OutputItemList::const_iterator i = items.begin(), iEnd = items.end(); i != iEnd; ++i) {

      ProductID const& id = i->branchDescription_->productID_;
      if (id == ProductID()) {
	throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	  << "PoolOutputModule::write: invalid ProductID supplied in productRegistry\n";
      }

      bool getProd = i->selected_ && (i->branchDescription_->produced() || i->renamed_ || !fastCloning);
      bool getProv = i->branchDescription_->produced() || i->renamed_ || !fastMetaCloning;

      EDProduct const* product = 0;
      BasicHandle const bh = principal.getForOutput(id, getProd, getProv);
      if (getProv) {
        if (bh.provenance() == 0) {
	  // No product with this ID is in the event.
	  // Create and write the provenance.
	  if (i->branchDescription_->produced_) {
	    EntryDescription entryDescription;
	    entryDescription.moduleDescriptionID_ = i->branchDescription_->moduleDescriptionID_;
            EntryDescriptionRegistry::instance()->insertMapped(entryDescription);
	    *i->entryDescriptionIDPtr_ = entryDescription.id();
	  } else {
	    throw edm::Exception(errors::ProductNotFound,"NoMatch")
	      << "PoolOutputModule: Unexpected internal error.  Contact the framework group.\n"
	      << "No group for branch" << i->branchDescription_->branchName_ << '\n';
	  }
        } else {
	  product = bh.wrapper();
	  *i->entryDescriptionIDPtr_ = bh.provenance()->event().id();
        }
      }
      if (getProd) {
	if (product == 0) {
	  // No product with this ID is in the event.
	  // Add a null product.
	  TClass *cp = gROOT->GetClass(i->branchDescription_->wrappedName().c_str());
	  product = static_cast<EDProduct *>(cp->New());
	}
	i->product_ = product;
      }
    }
    treePointers_[branchType]->fillTree();
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
}
