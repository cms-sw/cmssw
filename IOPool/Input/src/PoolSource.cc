/*----------------------------------------------------------------------
$Id: PoolSource.cc,v 1.12 2005/12/01 22:35:23 wmtan Exp $
----------------------------------------------------------------------*/

#include "IOPool/Input/src/PoolSource.h"
#include "IOPool/Common/interface/PoolNames.h"
#include "IOPool/Common/interface/ClassFiller.h"
#include "IOPool/Common/interface/RefStreamer.h"

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/EventAux.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

#include <stdexcept>

using std::auto_ptr;

#include <iostream>

namespace edm {
  PoolRASource::PoolRASource(ParameterSet const& pset, InputSourceDescription const& desc) :
    RandomAccessInputSource(desc),
    catalog_(PoolCatalog::READ,
      PoolCatalog::toPhysical(pset.getUntrackedParameter("catalog", std::string()))),
    productMap_(),
    file_(pset.getUntrackedParameter("fileName", std::string())),
    files_(pset.getUntrackedParameter("fileNames", std::vector<std::string>())),
    fileIter_(files_.begin()),
    poolFile_(),
    remainingEvents_(pset.getUntrackedParameter<int>("maxEvents", -1)),
    eventID_() {
    ClassFiller();
    if (file_.empty()) {
      if (files_.empty()) { // this will throw;
        pset.getUntrackedParameter<std::string>("fileName");
      } else {
        init(*fileIter_);
        ++fileIter_;
      }
    } else {
      init(file_);
    }
  }

  void PoolRASource::init(std::string const& file) {

    std::string pfn;
    catalog_.findFile(pfn, file);

    poolFile_ = boost::shared_ptr<PoolFile>(new PoolFile(pfn));
    if (poolFile_->productRegistry().nextID() > preg_->nextID()) {
      preg_->setNextID(poolFile_->productRegistry().nextID());
    }
    ProductRegistry::ProductList const& prodList = poolFile_->productRegistry().productList();
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin();
        it != prodList.end(); ++it) {
      preg_->copyProduct(it->second);
    }

    for (ProductRegistry::ProductList::const_iterator it = preg_->productList().begin();
         it != preg_->productList().end(); ++it) {
      productMap_.insert(std::make_pair(it->second.productID_, it->second));
    }
  }

  bool PoolRASource::next() {
    if(poolFile_->next()) return true;
    if(fileIter_ == files_.end()) return false;

    // save the product registry from the current file, temporarily
    boost::shared_ptr<ProductRegistry const> pReg(poolFile_->productRegistrySharedPtr());

    // delete the old PoolFile.  The file will be closed.
    poolFile_.reset();
    
    std::string pfn;
    catalog_.findFile(pfn, *fileIter_);

    poolFile_ = boost::shared_ptr<PoolFile>(new PoolFile(pfn));
    // make sure the new product registry is identical to the old one
    if (*pReg != poolFile_->productRegistry()) {
      throw cms::Exception("MismatchedInput","PoolSource::next()")
        << "File " << *fileIter_ << "\nhas different product registry than previous files\n";
    }
    ++fileIter_;
    return next();
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
    if (remainingEvents_ == 0) {
      return auto_ptr<EventPrincipal>(0);
    }
    if (!next()) {
      return auto_ptr<EventPrincipal>(0);
    }
    --remainingEvents_;
    EventAux evAux;
    EventProvenance evProv;
    EventAux *pEvAux = &evAux;
    EventProvenance *pEvProv = &evProv;
    poolFile_->auxBranch()->SetAddress(&pEvAux);
    poolFile_->provBranch()->SetAddress(&pEvProv);
    poolFile_->auxBranch()->GetEntry(poolFile_->entryNumber());
    poolFile_->provBranch()->GetEntry(poolFile_->entryNumber());
    eventID_ = evAux.id_;
    // We're not done ... so prepare the EventPrincipal
    boost::shared_ptr<DelayedReader> store_(new PoolDelayedReader(poolFile_->entryNumber(), *this));
    auto_ptr<EventPrincipal> thisEvent(new EventPrincipal(evAux.id_, evAux.time_, *preg_, evAux.process_history_, store_));
    // Loop over provenance
    std::vector<BranchEntryDescription>::iterator pit = evProv.data_.begin();
    std::vector<BranchEntryDescription>::iterator pitEnd = evProv.data_.end();
    for (; pit != pitEnd; ++pit) {
      if (pit->status != BranchEntryDescription::Success) continue;
      // BEGIN These lines read all branches
      // TBranch *br = branches_.find(poolNames::keyName(*pit))->second;
      // br->SetAddress(p);
      // br->GetEntry(poolFile_->entryNumber());
      // auto_ptr<Provenance> prov(new Provenance(*pit));
      // prov->product = productMap_[prov.event.productID_];
      // auto_ptr<Group> g(new Group(auto_ptr<EDProduct>(p), prov));
      // END These lines read all branches
      // BEGIN These lines defer reading branches
      auto_ptr<Provenance> prov(new Provenance);
      prov->event = *pit;
      prov->product = productMap_[prov->event.productID_];
      auto_ptr<Group> g(new Group(prov));
      // END These lines defer reading branches
      thisEvent->addGroup(g);
    }

    return thisEvent;
  }

  auto_ptr<EventPrincipal>
  PoolRASource::read(EventID const& id) {
    // For now, don't support multiple runs.
    assert (id.run() == eventID_.run());
    // For now, assume EventID's are all there.
    EntryNumber offset = static_cast<long>(id.event()) - static_cast<long>(eventID_.event());
    poolFile_->entryNumber() += offset;
    return read();
  }

  void
  PoolRASource::skip(int offset) {
    poolFile_->entryNumber() += offset;
  }

//---------------------------------------------------------------------
  PoolRASource::PoolFile::PoolFile(std::string const& fileName) :
    file_(fileName),
    entryNumber_(-1),
    entries_(0),
    productRegistry_(new ProductRegistry),
    branches_(),
    auxBranch_(0),
    provBranch_(0),
    filePtr_(0) {

    LogInfo("FwkJob") << "Input file " << file_ << " is about to be opened.";
    filePtr_ = TFile::Open(file_.c_str());
    if (filePtr_ == 0) {
      throw cms::Exception("FileNotFound","PoolRASource::PoolFile::PoolFile()")
        << "File " << file_ << " was not found.\n";
    }
    LogInfo("FwkJob") << "Input file " << file_ << " has been opened successfully.";

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
    }

  }

  PoolRASource::PoolFile::~PoolFile() {
    filePtr_->Close();
  }

  PoolRASource::PoolDelayedReader::~PoolDelayedReader() {}

  auto_ptr<EDProduct>
  PoolRASource::PoolDelayedReader::get(BranchKey const& k, EventPrincipal const* ep) const {
    SetRefStreamer(ep);
    TBranch *br = branches().find(k)->second.second;
    TClass *cp = gROOT->GetClass(branches().find(k)->second.first.c_str());
    auto_ptr<EDProduct> p(static_cast<EDProduct *>(cp->New()));
    EDProduct *pp = p.get();
    br->SetAddress(&pp);
    br->GetEntry(entryNumber_);
    return p;
  }
}
