/*----------------------------------------------------------------------
$Id: RootFile.cc,v 1.4 2006/02/08 00:44:28 wmtan Exp $
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TFile.h"
#include "TTree.h"

namespace edm {
//---------------------------------------------------------------------
  RootFile::RootFile(std::string const& fileName) :
    file_(fileName),
    eventID_(),
    entryNumber_(-1),
    entries_(0),
    productRegistry_(new ProductRegistry),
    branches_(),
    auxBranch_(0),
    provBranch_(0),
    filePtr_(0) {

    LogInfo("FwkJob") << "Input file is about to be opened.";
    LogInfo("FwkJob") << "PFN: " << file_;
    filePtr_ = TFile::Open(file_.c_str());
    if (filePtr_ == 0) {
      throw cms::Exception("FileNotFound","RootFile::RootFile()")
        << "File " << file_ << " was not found.\n";
    }
    LogInfo("FwkJob") << "Input file opened successfully.";

    TTree *metaDataTree = dynamic_cast<TTree *>(filePtr_->Get(poolNames::metaDataTreeName().c_str()));
    assert(metaDataTree != 0);

    // Load streamers for product dictionary and member/base classes.
    ProductRegistry *ppReg = productRegistry_.get();
    metaDataTree->SetBranchAddress(poolNames::productDescriptionBranchName().c_str(),(&ppReg));
    metaDataTree->GetEntry(0);

    TTree *eventTree = dynamic_cast<TTree *>(filePtr_->Get(poolNames::eventTreeName().c_str()));
    assert(eventTree != 0);
    entries_ = eventTree->GetEntries();

    auxBranch_ = eventTree->GetBranch(poolNames::auxiliaryBranchName().c_str());
    provBranch_ = eventTree->GetBranch(poolNames::provenanceBranchName().c_str());

    std::string const wrapperBegin("edm::Wrapper<");
    std::string const wrapperEnd1(">");
    std::string const wrapperEnd2(" >");

    ProductRegistry::ProductList const& prodList = productRegistry().productList();
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin();
        it != prodList.end(); ++it) {
      BranchDescription const& prod = it->second;
      prod.init();
      TBranch * branch = eventTree->GetBranch(prod.branchName_.c_str());
      std::string const& name = prod.fullClassName_;
      std::string const& wrapperEnd = (name[name.size()-1] == '>' ? wrapperEnd2 : wrapperEnd1);
      std::string const className = wrapperBegin + name + wrapperEnd;
      branches_.insert(std::make_pair(it->first, std::make_pair(className, branch)));
      productMap_.insert(std::make_pair(it->second.productID_, it->second));
    }
  }

  RootFile::~RootFile() {
    filePtr_->Close();
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
    boost::shared_ptr<DelayedReader> store_(new RootDelayedReader(entryNumber(), branches_));
    std::auto_ptr<EventPrincipal> thisEvent(new EventPrincipal(evAux.id_, evAux.time_, pReg, evAux.process_history_, store_));
    // Loop over provenance
    std::vector<BranchEntryDescription>::iterator pit = evProv.data_.begin();
    std::vector<BranchEntryDescription>::iterator pitEnd = evProv.data_.end();
    for (; pit != pitEnd; ++pit) {
      if (pit->status != BranchEntryDescription::Success) continue;
      // BEGIN These lines read all branches
      // TBranch *br = branches_.find(poolNames::keyName(*pit))->second;
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
    return thisEvent;
  }
}
