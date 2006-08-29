/*----------------------------------------------------------------------
$Id: RootFile.cc,v 1.29 2006/08/28 22:32:33 wmtan Exp $
----------------------------------------------------------------------*/

#include "IOPool/Input/src/RootFile.h"
#include "IOPool/Input/src/RootDelayedReader.h"
#include "FWCore/Utilities/interface/PersistentNames.h"

#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/BranchEntryDescription.h"
#include "DataFormats/Common/interface/EventAux.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "DataFormats/Common/interface/ParameterSetBlob.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Common/interface/ProcessHistoryRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include "TTree.h"
#include "Rtypes.h"

namespace edm {
//---------------------------------------------------------------------
  RootFile::RootFile(std::string const& fileName, std::string const& catalogName) :
    file_(fileName),
    catalog_(catalogName),
    branchNames_(),
    eventProvenance_(),
    eventProvenancePtrs_(),
    reportToken_(0),
    eventID_(),
    entryNumber_(-1),
    entries_(0),
    productRegistry_(new ProductRegistry),
    branches_(new BranchMap),
    productMap_(),
    eventTree_(0),
    eventMetaTree_(0),
    auxBranch_(0),
    filePtr_(TFile::Open(file_.c_str())) {

    open();

    // Set up buffers for registries.
    ProductRegistry *ppReg = productRegistry_.get();
    typedef std::map<ParameterSetID, ParameterSetBlob> PsetMap;
    PsetMap psetMap;
    ProcessHistoryMap pHistMap;
    ModuleDescriptionMap mdMap;
    PsetMap *psetMapPtr = &psetMap;
    ProcessHistoryMap *pHistMapPtr = &pHistMap;
    ModuleDescriptionMap *mdMapPtr = &mdMap;

    // Read the metadata tree.
    TTree *metaDataTree = dynamic_cast<TTree *>(filePtr_->Get(poolNames::metaDataTreeName().c_str()));
    assert(metaDataTree != 0);
    metaDataTree->SetMaxTreeSize(kMaxLong64);

    metaDataTree->SetBranchAddress(poolNames::productDescriptionBranchName().c_str(),(&ppReg));
    metaDataTree->SetBranchAddress(poolNames::parameterSetMapBranchName().c_str(), &psetMapPtr);
    metaDataTree->SetBranchAddress(poolNames::processHistoryMapBranchName().c_str(), &pHistMapPtr);
    metaDataTree->SetBranchAddress(poolNames::moduleDescriptionMapBranchName().c_str(), &mdMapPtr);

    metaDataTree->GetEntry(0);
    productRegistry().setFrozen();

    // Merge into the registries. For now, we do NOT merge the product registry.
    pset::Registry& psetRegistry = *pset::Registry::instance();
    for (PsetMap::const_iterator i = psetMap.begin(); i != psetMap.end(); ++i) {
      psetRegistry.insertMapped(ParameterSet(i->second.pset_));
    } 
    ProcessHistoryRegistry & processNameListRegistry = *ProcessHistoryRegistry::instance();
    for (ProcessHistoryMap::const_iterator j = pHistMap.begin(); j != pHistMap.end(); ++j) {
      processNameListRegistry.insertMapped(j->second);
    } 
    ModuleDescriptionRegistry & moduleDescriptionRegistry = *ModuleDescriptionRegistry::instance();
    for (ModuleDescriptionMap::const_iterator k = mdMap.begin(); k != mdMap.end(); ++k) {
      moduleDescriptionRegistry.insertMapped(k->second);
    } 

    // Read the event and event meta trees.
    eventTree_ = dynamic_cast<TTree *>(filePtr_->Get(poolNames::eventTreeName().c_str()));
    eventTree_->SetMaxTreeSize(kMaxLong64);
    assert(eventTree_ != 0);
    eventMetaTree_ = dynamic_cast<TTree *>(filePtr_->Get(poolNames::eventMetaDataTreeName().c_str()));
    eventMetaTree_->SetMaxTreeSize(kMaxLong64);
    assert(eventMetaTree_ != 0);
    entries_ = eventTree_->GetEntries();
    assert(entries_ == eventMetaTree_->GetEntries());

    auxBranch_ = eventTree_->GetBranch(poolNames::auxiliaryBranchName().c_str());

    // Set up information from the product registry.
    ProductRegistry::ProductList const& prodList = productRegistry().productList();
    eventProvenance_.reserve(prodList.size());
    eventProvenancePtrs_.reserve(prodList.size());
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin();
        it != prodList.end(); ++it) {
      BranchDescription const& prod = it->second;
      prod.init();
      prod.provenancePresent_ = (eventMetaTree_->GetBranch(prod.branchName().c_str()) != 0);
      TBranch * branch = eventTree_->GetBranch(prod.branchName().c_str());
      prod.present_ = (branch != 0);
      if (prod.provenancePresent()) {
        std::string const &name = prod.className();
        std::string const className = wrappedClassName(name);
        if (branch != 0) branches_->insert(std::make_pair(it->first, std::make_pair(className, branch)));
        productMap_.insert(std::make_pair(it->second.productID(), it->second));
        branchNames_.push_back(prod.branchName());
        int n = eventProvenance_.size();
        eventProvenance_.push_back(BranchEntryDescription());
        eventProvenancePtrs_.push_back(&eventProvenance_[n]);
        eventMetaTree_->SetBranchAddress(prod.branchName().c_str(),(&eventProvenancePtrs_[n]));
      }
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

  RootFile::EntryNumber
  RootFile::getEntryNumber(EventID const& eventID) const {
    RootFile::EntryNumber index = eventTree_->GetEntryNumberWithIndex(eventID.run(), eventID.event());
    if (index < 0) index = eventTree_->GetEntryNumberWithBestIndex(eventID.run(), eventID.event()) + 1;
    if (index >= entries_) index = -1;
    return index;
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
    EventAux *pEvAux = &evAux;
    auxBranch_->SetAddress(&pEvAux);
    auxBranch_->GetEntry(entryNumber());
    eventMetaTree_->GetEntry(entryNumber());
    eventID_ = evAux.id();
    // We're not done ... so prepare the EventPrincipal
    boost::shared_ptr<DelayedReader> store_(new RootDelayedReader(entryNumber(), branches_, filePtr_));
    std::auto_ptr<EventPrincipal> thisEvent(new EventPrincipal(
		eventID_, evAux.time(), pReg, evAux.luminosityBlockID(),
		evAux.processHistoryID_, store_));
    // Loop over provenance
    std::vector<BranchEntryDescription>::iterator pit = eventProvenance_.begin();
    std::vector<BranchEntryDescription>::iterator pitEnd = eventProvenance_.end();
    for (; pit != pitEnd; ++pit) {
      // if (pit->creatorStatus() != BranchEntryDescription::Success) continue;
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
      std::auto_ptr<Group> g(new Group(prov, prov->event.isPresent()));
      // END These lines defer reading branches
      thisEvent->addGroup(g);
    }
    // report event read from file
    Service<JobReport> reportSvc;
    reportSvc->eventReadFromFile(reportToken_, eventID_);
    return thisEvent;
  }
}
