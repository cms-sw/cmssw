/*----------------------------------------------------------------------
$Id: RootFile.cc,v 1.38 2006/12/01 03:36:16 wmtan Exp $
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
//used for friendlyName translation
#include "FWCore/Framework/src/FriendlyName.h"

#include "TTree.h"
#include "Rtypes.h"

namespace edm {
//---------------------------------------------------------------------
  RootFile::RootFile(std::string const& fileName,
		     std::string const& catalogName,
		     std::string const& logicalFileName) :
    file_(fileName),
    logicalFile_(logicalFileName),
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
    TTree::SetMaxTreeSize(kMaxLong64);

    open();

    // Set up buffers for registries.
    //CDJ need to read to a temporary registry so we can do a translation of the BranchKeys
    ProductRegistry tempReg;
    ProductRegistry *ppReg = &tempReg;
    //ProductRegistry *ppReg = productRegistry_.get();
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

    metaDataTree->SetBranchAddress(poolNames::productDescriptionBranchName().c_str(),(&ppReg));
    metaDataTree->SetBranchAddress(poolNames::parameterSetMapBranchName().c_str(), &psetMapPtr);
    metaDataTree->SetBranchAddress(poolNames::processHistoryMapBranchName().c_str(), &pHistMapPtr);
    metaDataTree->SetBranchAddress(poolNames::moduleDescriptionMapBranchName().c_str(), &mdMapPtr);

    metaDataTree->GetEntry(0);
    //CDJ freeze our temporary
    tempReg.setFrozen();
    //productRegistry().setFrozen();

    //Do the translation from the persistent registry to the transient one
    std::map<std::string,std::string> newBranchToOldBranch;
    {
      const ProductRegistry::ProductList& prodList = tempReg.productList();
      for (ProductRegistry::ProductList::const_iterator it = prodList.begin();
           it != prodList.end(); ++it) {
        BranchDescription const& prod = it->second;
	//need to call init to cause the branch name to be recalculated
	prod.init();
        BranchDescription newBD(prod);
        newBD.friendlyClassName_ = friendlyname::friendlyName(newBD.className());

        newBD.init();
        productRegistry_->addProduct(newBD);
	newBranchToOldBranch[newBD.branchName()]=prod.branchName();
      }
      productRegistry().setFrozen();
    }
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
    assert(eventTree_ != 0);
    eventMetaTree_ = dynamic_cast<TTree *>(filePtr_->Get(poolNames::eventMetaDataTreeName().c_str()));
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
      //use the translated branch name 
      std::string branchName = newBranchToOldBranch[prod.branchName()];
      prod.provenancePresent_ = (eventMetaTree_->GetBranch(branchName.c_str()) != 0);
      TBranch * branch = eventTree_->GetBranch(branchName.c_str());
      prod.present_ = (branch != 0);
      if (prod.provenancePresent()) {
        std::string const &name = prod.className();
        std::string const className = wrappedClassName(name);
        if (branch != 0) branches_->insert(std::make_pair(it->first, std::make_pair(className, branch)));
        productMap_.insert(std::make_pair(it->second.productID(), it->second));
	//we want the new branch name for the JobReport
        branchNames_.push_back(prod.branchName());
        int n = eventProvenance_.size();
        eventProvenance_.push_back(BranchEntryDescription());
        eventProvenancePtrs_.push_back(&eventProvenance_[n]);
        eventMetaTree_->SetBranchAddress(branchName.c_str(),(&eventProvenancePtrs_[n]));
      }
    }
  }

  RootFile::~RootFile() {
  }

  void
  RootFile::open() {
    if (filePtr_ == 0) {
      throw cms::Exception("FileNotFound","RootFile::RootFile()")
        << "File " << file_ << " was not found or could not be opened.\n";
    }
    // Report file opened.
    std::string const label = "source";
    std::string moduleName = "PoolSource";
    Service<JobReport> reportSvc;
    reportToken_ = reportSvc->inputFileOpened(file_,
               logicalFile_,
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
  RootFile::getEntryNumber(EventID const& theEventID) const {
    RootFile::EntryNumber index = eventTree_->GetEntryNumberWithIndex(theEventID.run(), theEventID.event());
    if (index < 0) index = eventTree_->GetEntryNumberWithBestIndex(theEventID.run(), theEventID.event()) + 1;
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
		eventID_, evAux.time(), pReg,
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
      // bool const isPresent = prov->event.isPresent();
      // std::auto_ptr<Group> g(new Group(std::auto_ptr<EDProduct>(p), prov, isPresent));
      // END These lines read all branches
      // BEGIN These lines defer reading branches
      std::auto_ptr<Provenance> prov(new Provenance);
      prov->event = *pit;
      prov->product = productMap_[prov->event.productID_];
      bool const isPresent = prov->event.isPresent();
      std::auto_ptr<Group> g(new Group(prov, isPresent));
      // END These lines defer reading branches
      thisEvent->addGroup(g);
    }
    // report event read from file
    Service<JobReport> reportSvc;
    reportSvc->eventReadFromFile(reportToken_, eventID_);
    return thisEvent;
  }
}
