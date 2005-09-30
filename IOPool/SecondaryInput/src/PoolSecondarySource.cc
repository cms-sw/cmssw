/*----------------------------------------------------------------------
$Id: PoolSecondarySource.cc,v 1.1 2005/09/28 06:11:47 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/EventAux.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "IOPool/SecondaryInput/src/PoolSecondarySource.h"
#include "IOPool/CommonService/interface/PoolNames.h"
#include "IOPool/CommonService/interface/ClassFiller.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TTree.h"

#include <stdexcept>

using std::auto_ptr;

#include <iostream>

namespace edm {
  PoolSecondarySource::PoolSecondarySource(ParameterSet const& pset, InputSourceDescription const& desc) :
    SecondaryInputSource(desc),
    file_(pset.getUntrackedParameter<std::string>("fileName")),
    branches_(),
    auxBranch_(0),
    provBranch_(0),
    pReg_(new ProductRegistry) {
    init();
  }

  void PoolSecondarySource::init() {
    ClassFiller();

    std::string const wrapperBegin("edm::Wrapper<");
    std::string const wrapperEnd1(">");
    std::string const wrapperEnd2(" >");

    TFile *filePtr = TFile::Open(file_.c_str());
    assert(filePtr != 0);

    TTree *metaDataTree = dynamic_cast<TTree *>(filePtr->Get(poolNames::metaDataTreeName().c_str()));
    assert(metaDataTree != 0);

    // Load streamers for product dictionary and member/base classes.
    ProductRegistry *ppReg = pReg_.get();
    metaDataTree->SetBranchAddress(poolNames::productDescriptionBranchName().c_str(),(&ppReg));
    metaDataTree->GetEntry(0);

    TTree *eventTree = dynamic_cast<TTree *>(filePtr->Get(poolNames::eventTreeName().c_str()));
    assert(eventTree != 0);
    entries_ = eventTree->GetEntries();

    auxBranch_ = eventTree->GetBranch(poolNames::auxiliaryBranchName().c_str());
    provBranch_ = eventTree->GetBranch(poolNames::provenanceBranchName().c_str());

    ProductRegistry::ProductList const& prodList = pReg_->productList();
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin();
        it != prodList.end(); ++it) {
      ProductDescription const& prod = it->second;
      prod.init();
      pReg_->copyProduct(prod);
      TBranch * branch = eventTree->GetBranch(prod.branchName_.c_str());
      std::string const& name = prod.fullClassName_;
      std::string const& wrapperEnd = (name[name.size()-1] == '>' ? wrapperEnd2 : wrapperEnd1);
      std::string const className = wrapperBegin + name + wrapperEnd;
      branches_.insert(std::make_pair(it->first, std::make_pair(className, branch)));
    }

    for (ProductRegistry::ProductList::const_iterator it = pReg_->productList().begin();
         it != pReg_->productList().end(); ++it) {
      productMap.insert(std::make_pair(it->second.productID_, it->second));
    }
  }

  PoolSecondarySource::~PoolSecondarySource() {
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
  void
  PoolSecondarySource::read(int idx, int number, std::vector<EventPrincipal*>& result) {
    
    for (int entry = idx, i = 0; i < number; ++entry, ++i) {
      if (entry == entries_) entry = 0; // Wrap around
      EventAux evAux;
      EventProvenance evProv;
      EventAux *pEvAux = &evAux;
      EventProvenance *pEvProv = &evProv;
      auxBranch_->SetAddress(&pEvAux);
      provBranch_->SetAddress(&pEvProv);
      auxBranch_->GetEntry(entry);
      provBranch_->GetEntry(entry);
      // We're not done ... so prepare the EventPrincipal
      boost::shared_ptr<DelayedReader> store_(new PoolDelayedReader(entry, *this));
      EventPrincipal *thisEvent = new EventPrincipal(evAux.id_, evAux.time_, *pReg_, evAux.process_history_, store_);
      // Loop over provenance
      std::vector<Provenance>::iterator pit = evProv.data_.begin();
      std::vector<Provenance>::iterator pitEnd = evProv.data_.end();
      for (; pit != pitEnd; ++pit) {
        // BEGIN These lines read all branches
        // TBranch *br = branches_.find(poolNames::keyName(*pit))->second;
        // XXXX *p = QQQ;
        // br->SetAddress(p);
        // br->GetEntry(idx);
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
      result.push_back(thisEvent);
    }
  }

  PoolSecondarySource::PoolDelayedReader::~PoolDelayedReader() {}

  auto_ptr<EDProduct>
  PoolSecondarySource::PoolDelayedReader::get(BranchKey const& k) const {
    TBranch *br = branches().find(k)->second.second;
    TClass *cp = gROOT->GetClass(branches().find(k)->second.first.c_str());
    auto_ptr<EDProduct> p(static_cast<EDProduct *>(cp->New()));
    EDProduct *pp = p.get();
    br->SetAddress(&pp);
    br->GetEntry(entryNumber_);
    return p;
  }
}
