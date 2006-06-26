/*----------------------------------------------------------------------
$Id: RootFile.cc,v 1.19 2006/06/24 05:46:06 wmtan Exp $
----------------------------------------------------------------------*/

#include "IOPool/Input/src/RootFile.h"
#include "IOPool/Input/src/RootDelayedReader.h"
#include "IOPool/Common/interface/PoolNames.h"

#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/BranchEntryDescription.h"
#include "DataFormats/Common/interface/EventAux.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Common/interface/EventProvenance.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "DataFormats/Common/interface/ParameterSetBlob.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "TTree.h"

namespace edm {
//---------------------------------------------------------------------
  RootFile::RootFile(std::string const& fileName, std::string const& catalogName) :
    file_(fileName),
    catalog_(catalogName),
    branchNames_(),
    reportToken_(0),
    eventID_(),
    entryNumber_(-1),
    entries_(0),
    productRegistry_(new ProductRegistry),
    branches_(new BranchMap),
    eventTree_(0),
    auxBranch_(0),
    provBranch_(0),
    filePtr_(TFile::Open(file_.c_str())) {

    open();

    TTree *metaDataTree = dynamic_cast<TTree *>(filePtr_->Get(poolNames::metaDataTreeName().c_str()));
    assert(metaDataTree != 0);

    // Load streamers for product dictionary and member/base classes.
    ProductRegistry *ppReg = productRegistry_.get();
    metaDataTree->SetBranchAddress(poolNames::productDescriptionBranchName().c_str(),(&ppReg));
    metaDataTree->GetEntry(0);
    productRegistry().setFrozen();

    eventTree_ = dynamic_cast<TTree *>(filePtr_->Get(poolNames::eventTreeName().c_str()));
    assert(eventTree_ != 0);
    entries_ = eventTree_->GetEntries();

    auxBranch_ = eventTree_->GetBranch(poolNames::auxiliaryBranchName().c_str());
    provBranch_ = eventTree_->GetBranch(poolNames::provenanceBranchName().c_str());

    ProductRegistry::ProductList const& prodList = productRegistry().productList();
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin();
        it != prodList.end(); ++it) {
      BranchDescription const& prod = it->second;
      prod.init();
      TBranch * branch = eventTree_->GetBranch(prod.branchName().c_str());
      std::string const &name = prod.className();
      std::string const className = wrappedClassName(name);
      branches_->insert(std::make_pair(it->first, std::make_pair(className, branch)));
      productMap_.insert(std::make_pair(it->second.productID(), it->second));
      branchNames_.push_back(prod.branchName());
    }
  }

  RootFile::~RootFile() {
  }

  void
  RootFile::open() {
    if (filePtr_ == 0) {
      throw cms::Exception("FileNotFound","RootFile::RootFile()")
        << "File " << file_ << " was not found.\n";
    }
    // Report file opened.
    std::string const label = "source";
    std::string moduleName = "PoolSource";
    std::string logicalFileName = "";
    Service<JobReport> reportSvc;
    reportToken_ = reportSvc->inputFileOpened(file_,
               logicalFileName,
               catalog_,
               moduleName,
               label,
               branchNames_); 
  }

  void
  RootFile::close() {
    // Do not close the TFile explicitly because a delayed reader may still be using it.
    // The shared pointers will take care of closing and deleting it.
    Service<JobReport> reportSvc;
    reportSvc->inputFileClosed(reportToken_);
  }

  void
  RootFile::fillParameterSetRegistry(pset::Registry & psetRegistry) const {
    ParameterSetID psetID;
    ParameterSetBlob psetBlob;
    ParameterSetID *psetIDptr = &psetID;
    ParameterSetBlob *psetBlobptr = &psetBlob;
    TTree *parameterSetTree = dynamic_cast<TTree *>(filePtr_->Get(poolNames::parameterSetTreeName().c_str()));
    if (!parameterSetTree) {
      return;
    }
    int nEntries = parameterSetTree->GetEntries();
    parameterSetTree->SetBranchAddress(poolNames::parameterSetIDBranchName().c_str(), &psetIDptr);
    parameterSetTree->SetBranchAddress(poolNames::parameterSetBranchName().c_str(), &psetBlobptr);
    for (int i = 0; i < nEntries; ++i) {
      parameterSetTree->GetEntry(i);
      psetRegistry.insertParameterSet(ParameterSet(psetBlob.pset_));
    }
  }

  RootFile::EntryNumber
  RootFile::getEntryNumber(EventID const& eventID) const {
    return eventTree_->GetEntryNumberWithIndex(eventID.run(), eventID.event());
  }

  // read() is responsible for creating, and setting up, the
  // EventPrincipal.
  //
  //   1. create an EventPrincipal with a unique EventID
  //   2. For each entry in the provenance, put in one Group,
  //      holding the Provenance for the corresponding EDProduct.
  //   3. set up the caches in the EventPrincipal to know about this
  //      Group.
  //
  // We do *not* create the EDProduct instance (the equivalent of reading
  // the branch containing this EDProduct. That will be done by the Delayed Reader,
  //  when it is asked to do so.
  //
  std::auto_ptr<EventPrincipal>
  RootFile::read(ProductRegistry const& pReg) {
    EventAux evAux;
    EventProvenance evProv;
    EventAux *pEvAux = &evAux;
    EventProvenance *pEvProv = &evProv;
    auxBranch_->SetAddress(&pEvAux);
    provBranch_->SetAddress(&pEvProv);
    auxBranch_->GetEntry(entryNumber());
    provBranch_->GetEntry(entryNumber());
    eventID_ = evAux.id_;
    // We're not done ... so prepare the EventPrincipal
    boost::shared_ptr<DelayedReader> store_(new RootDelayedReader(entryNumber(), branches_, filePtr_));
    std::auto_ptr<EventPrincipal> thisEvent(new EventPrincipal(evAux.id(), evAux.time(), pReg, evAux.processHistory(), store_));
    // Loop over provenance
    std::vector<BranchEntryDescription>::iterator pit = evProv.data_.begin();
    std::vector<BranchEntryDescription>::iterator pitEnd = evProv.data_.end();
    for (; pit != pitEnd; ++pit) {
      if (pit->status != BranchEntryDescription::Success) continue;
      // BEGIN These lines read all branches
      // TBranch *br = branches_->find(poolNames::keyName(*pit))->second;
      // br->SetAddress(p);
      // br->GetEntry(rootFile_->entryNumber());
      // std::auto_ptr<Provenance> prov(new Provenance(*pit));
      // prov->product = productMap_[prov.event.productID_];
      // std::auto_ptr<Group> g(new Group(std::auto_ptr<EDProduct>(p), prov));
      // END These lines read all branches
      // BEGIN These lines defer reading branches
      std::auto_ptr<Provenance> prov(new Provenance);
      prov->event = *pit;
      prov->product = productMap_[prov->event.productID_];
      std::auto_ptr<Group> g(new Group(prov));
      // END These lines defer reading branches
      thisEvent->addGroup(g);
    }
    // report event read from file
    Service<JobReport> reportSvc;
    reportSvc->eventReadFromFile(reportToken_, eventID_);
    return thisEvent;
  }
}
