// $Id: PoolOutputModule.cc,v 1.1 2005/11/01 22:53:40 wmtan Exp $
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "IOPool/Common/interface/PoolDataSvc.h"
#include "IOPool/Output/src/PoolOutputModule.h"

#include "DataSvc/Ref.h"
#include "DataSvc/IDataSvc.h"
#include "PersistencySvc/ITransaction.h"
#include "PersistencySvc/ISession.h"
#include "StorageSvc/DbType.h"

#include "TTree.h"

#include <vector>
#include <string>
#include <iomanip>

using namespace std;

namespace edm {
  PoolOutputModule::PoolOutputModule(ParameterSet const& pset) :
    OutputModule(pset.getUntrackedParameter("select", ParameterSet())),
    catalog_(PoolCatalog::WRITE,
      PoolCatalog::toPhysical(pset.getUntrackedParameter("catalog", std::string()))),
    context_(catalog_, true, false),
    fileName_(PoolCatalog::toPhysical(pset.getUntrackedParameter<string>("fileName"))),
    logicalFileName_(pset.getUntrackedParameter("logicalFileName", std::string())),
    commitInterval_(pset.getUntrackedParameter("commitInterval", 1000U)),
    maxFileSize_(pset.getUntrackedParameter<int>("maxSize", 0x7f000000)),
    fileCount_(0) {
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

  PoolOutputModule::PoolFile::PoolFile(PoolOutputModule *om) :
      outputItemList_(), file_(), lfn_(), eventCount_(0), fileSizeCheckEvent_(100),
      provenancePlacement_(), auxiliaryPlacement_(), productDescriptionPlacement_(),
      om_(om) {
    std::string const suffix(".root");
    std::string::size_type offset = om->fileName_.rfind(suffix);
    bool ext = (offset == om->fileName_.size() - suffix.size());
    std::string fileBase(ext ? om->fileName_.substr(0, offset): om->fileName_);
    if (om->fileCount_) {
      std::ostringstream ofilename;
      ofilename << fileBase << setw(3) << setfill('0') << om->fileCount_ - 1 << suffix;
      file_ = ofilename.str();
      if (!om->logicalFileName_.empty()) {
        std::ostringstream lfilename;
        lfilename << om->logicalFileName_ << setw(3) << setfill('0') << om->fileCount_ - 1;
        lfn_ = lfilename.str();
      }
    } else {
      file_ = fileBase + suffix;
      lfn_ = om->logicalFileName_;
    }
    makePlacement(poolNames::eventTreeName(), poolNames::provenanceBranchName(), provenancePlacement_);
    makePlacement(poolNames::eventTreeName(), poolNames::auxiliaryBranchName(), auxiliaryPlacement_);
    makePlacement(poolNames::metaDataTreeName(), poolNames::productDescriptionBranchName(), productDescriptionPlacement_);
    ProductRegistry pReg;
    pReg.setNextID(om->nextID_);
    for (Selections::const_iterator it = om->descVec_.begin();
      it != om->descVec_.end(); ++it) {
      pReg.copyProduct(**it);
      pool::Placement placement;
      makePlacement(poolNames::eventTreeName(), (*it)->branchName_, placement);
      outputItemList_.push_back(std::make_pair(*it, placement));
    }
    startTransaction();
    pool::Ref<ProductRegistry const> rp(om->context(), &pReg);
    rp.markWrite(productDescriptionPlacement_);
    commitAndFlushTransaction();
    om->catalog_.registerFile(file_, lfn_);
  }

  void PoolOutputModule::PoolFile::startTransaction() const {
    context()->transaction().start(pool::ITransaction::UPDATE);
  }

  void PoolOutputModule::PoolFile::commitTransaction() const {
    context()->transaction().commitAndHold();
  }

  void PoolOutputModule::PoolFile::commitAndFlushTransaction() const {
    context()->transaction().commit();
  }

  void PoolOutputModule::PoolFile::makePlacement(std::string const& treeName_, std::string const& branchName_, pool::Placement& placement) {
    placement.setTechnology(pool::ROOTTREE_StorageType.type());
    placement.setDatabase(file_, pool::DatabaseSpecification::PFN);
    placement.setContainerName(poolNames::containerName(treeName_, branchName_));
  }

  bool PoolOutputModule::PoolFile::writeOne(EventPrincipal const& e) {
    ++eventCount_;
    startTransaction();
    // Write auxiliary branch
    EventAux aux;
    aux.process_history_ = e.processHistory();
    aux.id_ = e.id();

    pool::Ref<const EventAux> ra(context(), &aux);
    ra.markWrite(auxiliaryPlacement_);	

    EventProvenance eventProvenance;
    // Loop over EDProduct branches, fill the provenance, and write the branch.
    for (OutputItemList::const_iterator i = outputItemList_.begin();
         i != outputItemList_.end(); ++i) {
      ProductID const& id = i->first->productID_;

      if (id == ProductID()) {
        throw edm::Exception(edm::errors::ProductNotFound,"InvalidID")
  	  << "PoolOutputModule::write: invalid ProductID supplied in productRegistry\n";
      }
      EventPrincipal::SharedGroupPtr const g = e.getGroup(id);
      if (g.get() == 0) {
        // No product with this ID is in the event.  Add a null one.
        BranchEntryDescription event;
        event.status = BranchEntryDescription::CreatorNotRun;
        event.productID_ = id;
        eventProvenance.data_.push_back(event);
        pool::Ref<EDProduct const> ref(context(), i->first->productPtr_);
        ref.markWrite(i->second);
      } else {
        eventProvenance.data_.push_back(g->provenance().event);
        if (g->product() == 0) {
          // This is wasteful, as it de-serializes and re-serializes.
          // Replace it with something better.
          e.resolve_(*g);
        }
        pool::Ref<EDProduct const> ref(context(), g->product());
        ref.markWrite(i->second);
      }
    }
    // Write the provenance branch
    pool::Ref<EventProvenance const> rp(context(), &eventProvenance);
    rp.markWrite(provenancePlacement_);
	
    commitTransaction();
    if (eventCount_ % om_->commitInterval_ == 0) {
      commitAndFlushTransaction();
    }

    if (eventCount_ >= fileSizeCheckEvent_) {
	size_t size = om_->context_.getFileSize(file_);
	unsigned long eventSize = size/eventCount_;
	if (size + 2*eventSize >= om_->maxFileSize_) {
          endFile();
          return true;
        } else {
	  unsigned long increment = (om_->maxFileSize_ - size)/eventSize;
	  increment -= increment/8;	// Prevents overshoot
	  fileSizeCheckEvent_ = eventCount_ + increment;
        }
    }
    return false;
  }

  void PoolOutputModule::PoolFile::endFile() {
    commitAndFlushTransaction();
    om_->catalog_.commitCatalog();
    context()->session().disconnectAll();
    setBranchAliases();
  }


  // For now, we must use root directly to set branch aliases, since there is no way to do this in POOL
  // We do this after POOL has closed the file.
  void
  PoolOutputModule::PoolFile::setBranchAliases() const {
    TFile f(file_.c_str(), "update");
    // TFile f(filen_.c_str(), "update");
    TTree *t = dynamic_cast<TTree *>(f.Get(poolNames::eventTreeName().c_str()));
    if (t) {
      for (Selections::const_iterator it = om_->descVec_.begin();
        it != om_->descVec_.end(); ++it) {
        BranchDescription const& pd = **it;
        std::string const& full = pd.branchName_ + "obj";
        std::string const& alias = (pd.productInstanceName_.empty() ? pd.module.moduleLabel_ : pd.productInstanceName_);
        t->SetAlias(alias.c_str(), full.c_str());
      }
      t->Write(t->GetName(), TObject::kWriteDelete);
    }
    f.Purge();
    f.Close();
  }
}
