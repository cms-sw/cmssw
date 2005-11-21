// $Id: PoolOutputModule.cc,v 1.1 2005/11/01 22:53:40 wmtan Exp $
#include "DataSvc/Ref.h"
#include "DataSvc/IDataSvc.h"
#include "StorageSvc/DbType.h"
#include "PersistencySvc/ITransaction.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/ITechnologySpecificAttributes.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "IOPool/Output/src/PoolOutputModule.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TTree.h"
#include <vector>
#include <string>

using namespace std;

namespace edm {
  PoolOutputModule::PoolOutputModule(ParameterSet const& pset) :
    OutputModule(pset.getUntrackedParameter("select", ParameterSet())),
    catalog_(PoolCatalog::WRITE,
      PoolCatalog::toPhysical(pset.getUntrackedParameter("catalog", std::string()))),
    context_(catalog_.createContext(true, false)),
    file_(PoolCatalog::toPhysical(pset.getUntrackedParameter<string>("fileName"))),
    lfn_(pset.getUntrackedParameter("logicalFileName", std::string())),
    commitInterval_(pset.getUntrackedParameter("commitInterval", 1000U)),
    eventCount_(0),
    provenancePlacement_(),
    auxiliaryPlacement_(),
    productDescriptionPlacement_() {
    makePlacement(poolNames::eventTreeName(), poolNames::provenanceBranchName(), provenancePlacement_);
    makePlacement(poolNames::eventTreeName(), poolNames::auxiliaryBranchName(), auxiliaryPlacement_);
    makePlacement(poolNames::metaDataTreeName(), poolNames::productDescriptionBranchName(), productDescriptionPlacement_);
  }

  void PoolOutputModule::beginJob(EventSetup const&) {
    ProductRegistry pReg;
    pReg.setNextID(nextID_);
    for (Selections::const_iterator it = descVec_.begin();
      it != descVec_.end(); ++it) {
      pReg.copyProduct(**it);
      pool::Placement placement;
      makePlacement(poolNames::eventTreeName(), (*it)->branchName_, placement);
      outputItemList_.push_back(std::make_pair(*it, placement));
    }
    startTransaction();
    pool::Ref<ProductRegistry const> rp(context_, &pReg);
    rp.markWrite(productDescriptionPlacement_);
    commitAndFlushTransaction();
    catalog_.registerFile(file_, lfn_);
  }

  void PoolOutputModule::endJob() {
    commitAndFlushTransaction();
    catalog_.commitCatalog();
    context_->session().disconnectAll();
    setBranchAliases();
  }

  PoolOutputModule::~PoolOutputModule() {
  }

  void PoolOutputModule::startTransaction() const {
    context_->transaction().start(pool::ITransaction::UPDATE);
  }

  void PoolOutputModule::commitTransaction() const {
    context_->transaction().commitAndHold();
  }

  void PoolOutputModule::commitAndFlushTransaction() const {
    context_->transaction().commit();
  }

  void PoolOutputModule::makePlacement(std::string const& treeName_, std::string const& branchName_, pool::Placement& placement) {
    placement.setTechnology(pool::ROOTTREE_StorageType.type());
    placement.setDatabase(file_, pool::DatabaseSpecification::PFN);
    placement.setContainerName(poolNames::containerName(treeName_, branchName_));
  }

  void PoolOutputModule::write(EventPrincipal const& e) {
    ++eventCount_;
    startTransaction();

    // Write auxiliary branch
    EventAux aux;
    aux.process_history_ = e.processHistory();
    aux.id_ = e.id();

    pool::Ref<const EventAux> ra(context_, &aux);
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
        pool::Ref<EDProduct const> ref(context_, i->first->productPtr_);
        ref.markWrite(i->second);
      } else {
        eventProvenance.data_.push_back(g->provenance().event);
        if (g->product() == 0) {
          // This is wasteful, as it de-serializes and re-serializes.
          // Replace it with something better.
          e.resolve_(*g);
        }
        pool::Ref<EDProduct const> ref(context_, g->product());
        ref.markWrite(i->second);
      }
    }
    // Write the provenance branch
    pool::Ref<EventProvenance const> rp(context_, &eventProvenance);
    rp.markWrite(provenancePlacement_);
	
    commitTransaction();
    if (eventCount_ % commitInterval_ == 0) {
      commitAndFlushTransaction();
    }
  }

  // For now, we must use root directly to set branch aliases, since there is no way to do this in POOL
  // We do this after POOL has closed the file.
  void
  PoolOutputModule::setBranchAliases() const {
    TFile f(file_.c_str(), "update");
    TTree *t = dynamic_cast<TTree *>(f.Get(poolNames::eventTreeName().c_str()));
    if (t) {
      for (Selections::const_iterator it = descVec_.begin();
        it != descVec_.end(); ++it) {
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
