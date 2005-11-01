/*----------------------------------------------------------------------
$Id: PoolSecondarySource.cc,v 1.9 2005/10/31 18:02:16 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/Framework/interface/EventAux.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "IOPool/SecondaryInput/src/PoolSecondarySource.h"
#include "IOPool/Common/interface/PoolNames.h"
#include "IOPool/Common/interface/ClassFiller.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "TTree.h"

#include <stdexcept>

using std::auto_ptr;

#include <iostream>

namespace edm {
  PoolSecondarySource::PoolSecondarySource(ParameterSet const& pset) :
    SecondaryInputSource(),
    catalog_(PoolCatalog::READ,
      PoolCatalog::toPhysical(pset.getUntrackedParameter("catalog", std::string()))),
    productMap_(),
    file_(pset.getUntrackedParameter<std::string>("fileName")),
//    file_(pset.getUntrackedParameter("fileName", std::string())),
//    files_(pset.getUntrackedParameter("fileNames", std::vector<std::string>())),
    files_(),
    fileIter_(files_.begin()),
    poolFile_(),
    pReg_(new ProductRegistry) {
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

  void PoolSecondarySource::init(std::string const& file) {
    ClassFiller();

    std::string pfn;
    catalog_.findFile(pfn, file);

    poolFile_ = boost::shared_ptr<PoolFile>(new PoolFile(pfn));
    ProductRegistry::ProductList const& prodList = poolFile_->productRegistry().productList();

    for (ProductRegistry::ProductList::const_iterator it = prodList.begin();
        it != prodList.end(); ++it) {
      productMap_.insert(std::make_pair(it->second.productID_, it->second));
    }
  }

  bool PoolSecondarySource::next() {
    if(poolFile_->next()) return true;

     
    if(files_.empty()) return false;
    if(file_.empty() && files_.size() == 1) return false;
    if(fileIter_ == files_.end() && files_.size() == 1) return false;
    if(fileIter_ == files_.end()) fileIter_ == files_.begin();

    poolFile_.reset();

    std::string pfn;
    catalog_.findFile(pfn, *fileIter_);

    poolFile_ = boost::shared_ptr<PoolFile>(new PoolFile(pfn));
    ++fileIter_;
    next();
    return false;
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
      if (!next()) entry = 0;
      EventAux evAux;
      EventProvenance evProv;
      EventAux *pEvAux = &evAux;
      EventProvenance *pEvProv = &evProv;
      poolFile_->auxBranch()->SetAddress(&pEvAux);
      poolFile_->provBranch()->SetAddress(&pEvProv);
      poolFile_->auxBranch()->GetEntry(entry);
      poolFile_->provBranch()->GetEntry(entry);
      // We're not done ... so prepare the EventPrincipal
      boost::shared_ptr<DelayedReader> store_(new PoolDelayedReader(entry, *this));
      EventPrincipal *thisEvent = new EventPrincipal(evAux.id_, evAux.time_, *pReg_, evAux.process_history_, store_);
      // Loop over provenance
      std::vector<BranchEntryDescription>::iterator pit = evProv.data_.begin();
      std::vector<BranchEntryDescription>::iterator pitEnd = evProv.data_.end();
      for (; pit != pitEnd; ++pit) {
        if (pit->status != BranchEntryDescription::Success) continue;
        // BEGIN These lines read all branches
        // TBranch *br = branches_.find(poolNames::keyName(*pit))->second;
        // XXXX *p = QQQ;
        // br->SetAddress(p);
        // br->GetEntry(idx);
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
      result.push_back(thisEvent);
    }
  }

//---------------------------------------------------------------------
  PoolSecondarySource::PoolFile::PoolFile(std::string const& fileName) :
    file_(fileName),
    entryNumber_(-1),
    entries_(0),
    productRegistry_(),
    branches_(),
    auxBranch_(0),
    provBranch_(0),
    filePtr_(0) {

    filePtr_ = TFile::Open(file_.c_str());
    if (filePtr_ == 0) {
      throw cms::Exception("FileNotFound","PoolSecondarySource::PoolFile::PoolFile()")
        << "File " << file_ << " was not found.\n";
    }

    TTree *metaDataTree = dynamic_cast<TTree *>(filePtr_->Get(poolNames::metaDataTreeName().c_str()));
    assert(metaDataTree != 0);

    // Load streamers for product dictionary and member/base classes.
    ProductRegistry *ppReg = &productRegistry_;
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
 
  PoolSecondarySource::PoolFile::~PoolFile() {
    filePtr_->Close();
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
