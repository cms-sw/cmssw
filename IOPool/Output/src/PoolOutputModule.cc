// $Id: PoolOutputModule.cc,v 1.59 2006/12/25 04:22:41 wmtan Exp $

#include "IOPool/Output/src/PoolOutputModule.h"
#include "IOPool/Common/interface/PoolDataSvc.h"
#include "IOPool/Common/interface/ClassFiller.h"
#include "IOPool/Common/interface/RefStreamer.h"
#include "IOPool/Common/interface/CustomStreamer.h"

#include "DataFormats/Common/interface/BranchKey.h"
#include "DataFormats/Common/interface/FileFormatVersion.h"
#include "FWCore/Utilities/interface/GetFileFormatVersion.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/Common/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Common/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/ParameterSetBlob.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/JobReport.h"

#include "DataSvc/Ref.h"
#include "DataSvc/IDataSvc.h"
#include "PersistencySvc/ITransaction.h"
#include "PersistencySvc/ISession.h"
#include "StorageSvc/DbType.h"

#include "TTree.h"
#include "TFile.h"
#include "Rtypes.h"

#include <map>
#include <vector>
#include <string>
#include <iomanip>

namespace edm {
  PoolOutputModule::PoolOutputModule(ParameterSet const& pset) :
    OutputModule(pset),
    catalog_(pset),
    context_(catalog_, false),
    commitInterval_(pset.getUntrackedParameter<unsigned int>("commitInterval", 100U)),
    maxFileSize_(pset.getUntrackedParameter<int>("maxSize", 0x7f000000)),
    compressionLevel_(pset.getUntrackedParameter<int>("compressionLevel", 1)),
    moduleLabel_(pset.getParameter<std::string>("@module_label")),
    fileCount_(0),
    poolFile_() {
    ClassFiller();
    // We need to set a custom streamer for edm::RefCore so that it will not be split.
    // even though a custom streamer is not otherwise necessary.
    SetRefStreamer();
    // We need to set a custom streamer for top level provenance objects so they will not be split,
    // This facilitates backward compatibility, since custom streamers can be used for reading
    // if the branch is not split.
    // Note: The map classes already have a custom streamer, so this is not needed now.
    // We do this only to protect against future root changes.
    typedef std::map<ParameterSetID, ParameterSetBlob> ParameterSetMap;
    SetCustomStreamer<EventAux>();
    SetCustomStreamer<ProductRegistry>();
    SetCustomStreamer<ParameterSetMap>();
    SetCustomStreamer<ProcessHistoryMap>();
    SetCustomStreamer<ModuleDescriptionMap>();
    SetCustomStreamer<FileFormatVersion>();
    SetCustomStreamer<BranchEntryDescription>();
    SetCustomStreamer<LuminosityBlockAux>();
    SetCustomStreamer<RunAux>();
  }

  void PoolOutputModule::beginJob(EventSetup const&) {
    poolFile_ = boost::shared_ptr<PoolFile>(new PoolFile(this));
  }

  void PoolOutputModule::endJob() {
    poolFile_->endFile();
  }

  PoolOutputModule::~PoolOutputModule() {
  }

  void PoolOutputModule::write(EventPrincipal const& e) {
      if (poolFile_->writeOne(e)) {
	++fileCount_;
	poolFile_ = boost::shared_ptr<PoolFile>(new PoolFile(this));
      }
  }

  void PoolOutputModule::endLuminosityBlock(LuminosityBlockPrincipal const& lb) {
      poolFile_->writeLuminosityBlock(lb);
  }

  void PoolOutputModule::endRun(RunPrincipal const& r) {
      poolFile_->writeRun(r);
  }

  PoolOutputModule::PoolFile::PoolFile(PoolOutputModule *om) :
    outputItemList_(), branchNames_(),
      file_(), lfn_(),
      reportToken_(0), eventCount_(0),
      fileSizeCheckEvent_(100),
      auxiliaryPlacement_(),
      productDescriptionPlacement_(),
      parameterSetPlacement_(),
      moduleDescriptionPlacement_(),
      processHistoryPlacement_(),
      fileFormatVersionPlacement_(),
      om_(om) {
    TTree::SetMaxTreeSize(kMaxLong64);
    std::string suffix(".root");
    std::string::size_type offset = om_->fileName().rfind(suffix);
    bool ext = (offset == om_->fileName().size() - suffix.size());
    if (!ext) suffix.clear();
    std::string fileBase(ext ? om_->fileName().substr(0, offset) : om_->fileName());
    if (om_->fileCount_) {
      std::ostringstream ofilename;
      ofilename << fileBase << std::setw(3) << std::setfill('0') << om_->fileCount_ << suffix;
      file_ = ofilename.str();
      if (!om_->logicalFileName().empty()) {
	std::ostringstream lfilename;
	lfilename << om_->logicalFileName() << std::setw(3) << std::setfill('0') << om_->fileCount_;
	lfn_ = lfilename.str();
      }
    } else {
      file_ = fileBase + suffix;
      lfn_ = om_->logicalFileName();
    }
    makePlacement(poolNames::metaDataTreeName(), poolNames::productDescriptionBranchName(),
	productDescriptionPlacement_);
    makePlacement(poolNames::metaDataTreeName(), poolNames::parameterSetMapBranchName(),
	parameterSetPlacement_);
    makePlacement(poolNames::metaDataTreeName(), poolNames::moduleDescriptionMapBranchName(),
	moduleDescriptionPlacement_);
    makePlacement(poolNames::metaDataTreeName(), poolNames::processHistoryMapBranchName(),
	processHistoryPlacement_);
    makePlacement(poolNames::metaDataTreeName(), poolNames::fileFormatVersionBranchName(),
	fileFormatVersionPlacement_);
   
    for (int i = InEvent; i < EndBranchType; ++i) {
      BranchType branchType = static_cast<BranchType>(i);
      std::string productTreeName = BranchTypeToProductTreeName(branchType);
      std::string metaDataTreeName = BranchTypeToMetaDataTreeName(branchType);
      OutputItemList & outputItemList = outputItemList_[branchType];
      std::vector<std::string> & branchNames = branchNames_[branchType];
      Selections const& descVec = om_->descVec_[branchType];
      Selections const& droppedVec = om_->droppedVec_[branchType];
      
      makePlacement(productTreeName, BranchTypeToAuxiliaryBranchName(branchType),
		    auxiliaryPlacement_[branchType]);
      for (Selections::const_iterator it = descVec.begin(); it != descVec.end(); ++it) {
        pool::Placement provenancePlacement;
        pool::Placement productPlacement;
        makePlacement(metaDataTreeName, (*it)->branchName(), provenancePlacement);
        makePlacement(productTreeName, (*it)->branchName(), productPlacement);
        outputItemList.push_back(OutputItem(*it, true, provenancePlacement, productPlacement));
        branchNames.push_back((*it)->branchName());
      }
      for (Selections::const_iterator it = droppedVec.begin(); it != droppedVec.end(); ++it) {
        pool::Placement provenancePlacement;
        makePlacement(metaDataTreeName, (*it)->branchName(), provenancePlacement);
        outputItemList.push_back(OutputItem(*it, false, provenancePlacement));
      }
    }

    pool::FileCatalog::FileID fid = om_->catalog_.registerFile(file_, lfn_);
    startTransaction();

    FileFormatVersion fileFormatVersion(edm::getFileFormatVersion());

    pool::Ref<FileFormatVersion const> fft(om_->context(), &fileFormatVersion);
    fft.markWrite(fileFormatVersionPlacement_);

    // Now, we can set the ROOT compression level
    om_->context_.setCompressionLevel(file_, om_->compressionLevel_);

    commitAndFlushTransaction();
    // Register the output file with the JobReport service
    // and get back the token for it.
    std::string moduleName = "PoolOutputModule";
    Service<JobReport> reportSvc;
    reportToken_ = reportSvc->outputFileOpened(
		      file_, lfn_,  // PFN and LFN
		      om_->catalog_.url(),  // catalog
		      moduleName,   // module class name
		      om_->moduleLabel_,  // module label
		      fid, // file id (guid)
		      branchNames_[InEvent]); // branch names being written
  }

  void PoolOutputModule::PoolFile::startTransaction() const {
   context()->transaction().start(pool::ITransaction::UPDATE);
  }

  void PoolOutputModule::PoolFile::commitTransaction() const {
    bool ret = context()->transaction().commitAndHold();
    if (!ret) {
      std::string message = "Fatal Pool Error in commitAndHoldTransaction.\n";
      Exception except(edm::errors::FatalRootError, message);
      throw except;
    }
    provenances_.clear();
  }

  void PoolOutputModule::PoolFile::commitAndFlushTransaction() const {
    bool ret = context()->transaction().commit();
    if (!ret) {
      std::string message = "Fatal Pool Error in commitTransaction.\n";
      Exception except(edm::errors::FatalRootError, message);
      throw except;
    }
    provenances_.clear();
  }

  void PoolOutputModule::PoolFile::makePlacement(std::string const& treeName_, std::string const& branchName, pool::Placement& placement) {
    placement.setTechnology(pool::ROOTTREE_StorageType.type());
    placement.setDatabase(file_, pool::DatabaseSpecification::PFN);
    placement.setContainerName(poolNames::containerName(treeName_, branchName));
  }

  bool PoolOutputModule::PoolFile::writeOne(EventPrincipal const& e) {
    ++eventCount_;
    startTransaction();
    // Write auxiliary branch
    EventAux aux;
    aux.processHistoryID_ = e.processHistoryID();
    aux.id_ = e.id();
    aux.time_ = e.time();

    pool::Ref<EventAux const> ra(context(), &aux);
    ra.markWrite(auxiliaryPlacement_[InEvent]);	

    if (!outputItemList_[InEvent].empty()) fillBranches(outputItemList_[InEvent], e.groupGetter());

    commitTransaction();

    // Report event written 
    Service<JobReport> reportSvc;
    reportSvc->eventWrittenToFile(reportToken_, e.id().run(), e.id().event());

    if (eventCount_ >= fileSizeCheckEvent_) {
	unsigned int const oneK = 1024;
	unsigned int size = om_->context_.getFileSize(file_)/oneK;
	unsigned int eventSize = std::max(size/eventCount_, 1U);
	if (size + 2*eventSize >= om_->maxFileSize_) {
	  endFile();
	  return true;
	} else {
	  unsigned int increment = (om_->maxFileSize_ - size)/eventSize;
	  increment -= increment/8;	// Prevents overshoot
	  fileSizeCheckEvent_ = eventCount_ + increment;
	}
    }
    if (eventCount_ % om_->commitInterval_ == 0) {
      commitAndFlushTransaction();
      startTransaction();
    }
    return false;
  }

  void
  PoolOutputModule::PoolFile::writeLuminosityBlock(LuminosityBlockPrincipal const& lb) {
    startTransaction();
    // Write auxiliary branch
    LuminosityBlockAux aux;
    aux.processHistoryID_ = lb.processHistoryID();
    aux.id_ = lb.id();
    pool::Ref<LuminosityBlockAux const> ra(context(), &aux);
    ra.markWrite(auxiliaryPlacement_[InLumi]);	
    if (!outputItemList_[InLumi].empty()) fillBranches(outputItemList_[InLumi], lb.groupGetter());
    commitTransaction();
  }

  void
  PoolOutputModule::PoolFile::writeRun(RunPrincipal const& r) {
    startTransaction();
    // Write auxiliary branch
    RunAux aux;
    aux.processHistoryID_ = r.processHistoryID();
    aux.id_ = r.id();
    pool::Ref<RunAux const> ra(context(), &aux);
    ra.markWrite(auxiliaryPlacement_[InRun]);	
    if (!outputItemList_[InRun].empty()) fillBranches(outputItemList_[InRun], r.groupGetter());
    commitTransaction();
  }

  void
  PoolOutputModule::PoolFile::fillBranches(OutputItemList const& items, DataBlockImpl const& dataBlock) const {

    // Loop over EDProduct branches, fill the provenance, and write the branch.
    for (OutputItemList::const_iterator i = items.begin();
	 i != outputItemList_[InEvent].end(); ++i) {
      ProductID const& id = i->branchDescription_->productID_;

      if (id == ProductID()) {
	throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
	  << "PoolOutputModule::write: invalid ProductID supplied in productRegistry\n";
      }

      EDProduct const* product = 0;
      DataBlockImpl::SharedConstGroupPtr const g = dataBlock.getGroup(id, i->selected_);
      if (g.get() == 0) {
	// No Group with this ID is in the event.
	// Create and write the provenance.
	if (i->branchDescription_->produced_) {
          BranchEntryDescription provenance;
	  provenance.moduleDescriptionID_ = i->branchDescription_->moduleDescriptionID_;
	  provenance.productID_ = id;
	  provenance.status_ = BranchEntryDescription::CreatorNotRun;
	  provenance.isPresent_ = false;
	  provenance.cid_ = 0;
	  
	  provenances_.push_front(provenance); 
          pool::Ref<BranchEntryDescription const> refp(context(), &*provenances_.begin());
          refp.markWrite(i->provenancePlacement_);
	} else {
	    throw edm::Exception(errors::ProductNotFound,"NoMatch")
	      << "PoolOutputModule: Unexpected internal error.  Contact the framework group.\n"
	      << "No group for branch" << i->branchDescription_->branchName_ << '\n';
	}
      } else {
	// There is a Group with this ID is in the event.  Write the provenance.
	bool present = i->selected_ && g->product() && g->product()->isPresent();
	if (present == g->provenance().event.isPresent()) {
	  // The provenance can be written out as is, saving a copy. 
          pool::Ref<BranchEntryDescription const> refp(context(), &g->provenance().event);
          refp.markWrite(i->provenancePlacement_);
	} else {
	  // We need to make a private copy of the provenance so we can set isPresent_ correctly.
	  provenances_.push_front(g->provenance().event);
	  provenances_.begin()->isPresent_ = present;
          pool::Ref<BranchEntryDescription const> refp(context(), &*provenances_.begin());
          refp.markWrite(i->provenancePlacement_);
	}
	product = g->product();
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
	pool::Ref<EDProduct const> ref(context(), product);
	ref.markWrite(i->productPlacement_);
      }
    }
  }

  void PoolOutputModule::PoolFile::endFile() {
    startTransaction();

    Service<ConstProductRegistry> reg;
    ProductRegistry pReg = reg->productRegistry();
    pool::Ref<ProductRegistry const> rp(om_->context(), &pReg);
    rp.markWrite(productDescriptionPlacement_);

    pool::Ref<ModuleDescriptionMap const> rmod(om_->context(), &ModuleDescriptionRegistry::instance()->data());
    rmod.markWrite(moduleDescriptionPlacement_);

    typedef std::map<ParameterSetID, ParameterSetBlob> ParameterSetMap;
    ParameterSetMap psetMap;
    pset::Registry const* psetRegistry = pset::Registry::instance();    
    for (pset::Registry::const_iterator it = psetRegistry->begin(); it != psetRegistry->end(); ++it) {
      psetMap.insert(std::make_pair(it->first, ParameterSetBlob(it->second.toStringOfTracked())));
    }
    pool::Ref<ParameterSetMap const> rpparam(om_->context(), &psetMap);
    rpparam.markWrite(parameterSetPlacement_);

    pool::Ref<ProcessHistoryMap const> rhist(om_->context(), &ProcessHistoryRegistry::instance()->data());
    rhist.markWrite(processHistoryPlacement_);

    commitAndFlushTransaction();
    om_->catalog_.commitCatalog();
    context()->session().disconnectAll();
    rootPostProcess();
    // report that file has been closed
    Service<JobReport> reportSvc;
    reportSvc->outputFileClosed(reportToken_);
  }


  // For now, we must use root directly to set Tree indices and branch aliases,
  // since there is no way to do this in POOL
  // We do this after POOL has closed the file.
  void
  PoolOutputModule::PoolFile::rootPostProcess() const {
    std::auto_ptr<TFile> pf(TFile::Open(file_.c_str(), "update"));
    TFile &f = *pf;
    TTree *tEvent = dynamic_cast<TTree *>(f.Get(BranchTypeToProductTreeName(InEvent).c_str()));
    if (tEvent) {
      tEvent->BuildIndex("id_.run_", "id_.event_");
      setBranchAliases(tEvent, om_->descVec_[InEvent]);
    }
    TTree *tLumi = dynamic_cast<TTree *>(f.Get(BranchTypeToProductTreeName(InLumi).c_str()));
    if (tLumi) {
      tLumi->BuildIndex("runID_", "id_");
      setBranchAliases(tLumi, om_->descVec_[InLumi]);
    }
    TTree *tRun = dynamic_cast<TTree *>(f.Get(BranchTypeToProductTreeName(InRun).c_str()));
    if (tRun) {
      tRun->BuildIndex("id_");
      setBranchAliases(tRun, om_->descVec_[InRun]);
    }
    f.Purge();
    f.Close();
  }
  
  void
  PoolOutputModule::PoolFile::setBranchAliases(TTree *tree, Selections const& branches) const {
    if (tree) {
      for (Selections::const_iterator i = branches.begin();
	  i != branches.end(); ++i) {
	BranchDescription const& pd = **i;
	std::string const& full = pd.branchName() + "obj";
	if (pd.branchAliases().empty()) {
	  std::string const& alias =
	      (pd.productInstanceName().empty() ? pd.moduleLabel() : pd.productInstanceName());
	  tree->SetAlias(alias.c_str(), full.c_str());
	} else {
	  std::set<std::string>::const_iterator it = pd.branchAliases().begin();
	  std::set<std::string>::const_iterator itend = pd.branchAliases().end();
	  for (; it != itend; ++it) {
	    tree->SetAlias((*it).c_str(), full.c_str());
	  }
	}
      }
      tree->Write(tree->GetName(), TObject::kWriteDelete);
    }
  }
}
