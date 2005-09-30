/*----------------------------------------------------------------------
$Id: PoolSource.cc,v 1.2 2005/09/30 05:01:37 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/EventAux.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "IOPool/Input/src/PoolSource.h"
#include "IOPool/CommonService/interface/PoolNames.h"
#include "IOPool/CommonService/interface/ClassFiller.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TTree.h"

#include <stdexcept>

using std::auto_ptr;

#include <iostream>

namespace edm {
  PoolRASource::PoolRASource(ParameterSet const& pset, InputSourceDescription const& desc) :
    RandomAccessInputSource(desc),
    file_(pset.getUntrackedParameter<std::string>("fileName")),
    remainingEvents_(pset.getUntrackedParameter<int>("maxEvents", -1)),
    entryNumber_(0),
    eventID_(),
    branches_(),
    auxBranch_(0),
    provBranch_(0) {
    init();
  }

  void PoolRASource::init() {
    ClassFiller();

    std::string const wrapperBegin("edm::Wrapper<");
    std::string const wrapperEnd1(">");
    std::string const wrapperEnd2(" >");

    TFile *filePtr = TFile::Open(file_.c_str());
    assert(filePtr != 0);

    TTree *metaDataTree = dynamic_cast<TTree *>(filePtr->Get(poolNames::metaDataTreeName().c_str()));
    assert(metaDataTree != 0);

    // Load streamers for product dictionary and member/base classes.
    ProductRegistry pReg;
    ProductRegistry *ppReg = &pReg;
    metaDataTree->SetBranchAddress(poolNames::productDescriptionBranchName().c_str(),(&ppReg));
    metaDataTree->GetEntry(0);

    TTree *eventTree = dynamic_cast<TTree *>(filePtr->Get(poolNames::eventTreeName().c_str()));
    assert(eventTree != 0);
    entries_ = eventTree->GetEntries();

    auxBranch_ = eventTree->GetBranch(poolNames::auxiliaryBranchName().c_str());
    provBranch_ = eventTree->GetBranch(poolNames::provenanceBranchName().c_str());

    if (pReg.nextID() > preg_->nextID()) {
      preg_->setNextID(pReg.nextID());
    }

    ProductRegistry::ProductList const& prodList = pReg.productList();
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin();
        it != prodList.end(); ++it) {
      ProductDescription const& prod = it->second;
      prod.init();
      preg_->copyProduct(prod);
      TBranch * branch = eventTree->GetBranch(prod.branchName_.c_str());
      std::string const& name = prod.fullClassName_;
      std::string const& wrapperEnd = (name[name.size()-1] == '>' ? wrapperEnd2 : wrapperEnd1);
      std::string const className = wrapperBegin + name + wrapperEnd;
      branches_.insert(std::make_pair(it->first, std::make_pair(className, branch)));
    }

    for (ProductRegistry::ProductList::const_iterator it = preg_->productList().begin();
         it != preg_->productList().end(); ++it) {
      productMap.insert(std::make_pair(it->second.productID_, it->second));
    }
  }

  PoolRASource::~PoolRASource() {
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
  auto_ptr<EventPrincipal>
  PoolRASource::read() {
    // If we're done, or out of range, return a null auto_ptr
    if (remainingEvents_ == 0 || entryNumber_ >= entries_ || entryNumber_ < 0) {
      return auto_ptr<EventPrincipal>(0);
    }
    --remainingEvents_;
    EventAux evAux;
    EventProvenance evProv;
    EventAux *pEvAux = &evAux;
    EventProvenance *pEvProv = &evProv;
    auxBranch_->SetAddress(&pEvAux);
    provBranch_->SetAddress(&pEvProv);
    auxBranch_->GetEntry(entryNumber_);
    provBranch_->GetEntry(entryNumber_);
    eventID_ = evAux.id_;
    // We're not done ... so prepare the EventPrincipal
    boost::shared_ptr<DelayedReader> store_(new PoolDelayedReader(entryNumber_, *this));
    auto_ptr<EventPrincipal> thisEvent(new EventPrincipal(evAux.id_, evAux.time_, *preg_, evAux.process_history_, store_));
    // Loop over provenance
    std::vector<Provenance>::iterator pit = evProv.data_.begin();
    std::vector<Provenance>::iterator pitEnd = evProv.data_.end();
    for (; pit != pitEnd; ++pit) {
      // BEGIN These lines read all branches
      // TBranch *br = branches_.find(poolNames::keyName(*pit))->second;
      // XXXX *p = QQQ;
      // br->SetAddress(p);
      // br->GetEntry(entryNumber_);
      // auto_ptr<Provenance> prov(new Provenance(*pit));
      // prov->product = productMap[prov.event.productID_];
      // auto_ptr<Group> g(new Group(auto_ptr<EDProduct>(p), prov));
      // END These lines read all branches
      // BEGIN These lines defer reading branches
      auto_ptr<Provenance> prov(new Provenance);
      prov->event = pit->event;
      prov->product = productMap[prov->event.productID_];
      auto_ptr<Group> g(new Group(prov));
      // END These lines defer reading branches
      thisEvent->addGroup(g);
    }

    ++entryNumber_;
    return thisEvent;
  }

  auto_ptr<EventPrincipal>
  PoolRASource::read(EventID const& id) {
    // For now, don't support multiple runs.
    assert (id.run() == eventID_.run());
    // For now, assume EventID's are all there.
    EntryNumber offset = static_cast<long>(id.event()) - static_cast<long>(eventID_.event());
    entryNumber_ += offset;
    return read();
  }

  void
  PoolRASource::skip(int offset) {
    entryNumber_ += offset;
  }

  PoolRASource::PoolDelayedReader::~PoolDelayedReader() {}

  auto_ptr<EDProduct>
  PoolRASource::PoolDelayedReader::get(BranchKey const& k) const {
    TBranch *br = branches().find(k)->second.second;
    TClass *cp = gROOT->GetClass(branches().find(k)->second.first.c_str());
    auto_ptr<EDProduct> p(static_cast<EDProduct *>(cp->New()));
    EDProduct *pp = p.get();
    br->SetAddress(&pp);
    br->GetEntry(entryNumber_);
    return p;
  }
}
