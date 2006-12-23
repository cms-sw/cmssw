/*----------------------------------------------------------------------
$Id: RootFile.cc,v 1.42 2006/12/14 04:30:59 wmtan Exp $
----------------------------------------------------------------------*/

#include "IOPool/Input/src/RootFile.h"
#include "IOPool/Input/src/RootDelayedReader.h"

#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/BranchEntryDescription.h"
#include "DataFormats/Common/interface/BranchType.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "DataFormats/Common/interface/ParameterSetBlob.h"
#include "DataFormats/Common/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Common/interface/ProcessHistoryRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
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
      filePtr_(TFile::Open(file_.c_str())),
      reportToken_(0),
      eventAux_(),
      lumiAux_(),
      runAux_(),
      eventTree_(*filePtr_, InEvent),
      lumiTree_(*filePtr_, InLumi),
      runTree_(*filePtr_, InRun),
      treePointers_(),
      productRegistry_(new ProductRegistry),
      branches_(new BranchMap),
      products_(),
      luminosityBlockPrincipal_() {
    treePointers_[InEvent] = &eventTree_;
    treePointers_[InLumi]  = &lumiTree_;
    treePointers_[InRun]   = &runTree_;

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

    // Set up information from the product registry.
    ProductRegistry::ProductList const& prodList = productRegistry().productList();
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin();
        it != prodList.end(); ++it) {
      BranchDescription const& prod = it->second;
      treePointers_[prod.branchType()]->addBranch(it->first, prod, *branches_, products_,
						 newBranchToOldBranch[prod.branchName()]);
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
               eventTree().branchNames()); 
  }

  void
  RootFile::close() {
    // Do not close the TFile explicitly because a delayed reader may still be using it.
    // The shared pointers will take care of closing and deleting it.
    Service<JobReport> reportSvc;
    reportSvc->inputFileClosed(reportToken_);
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
    TTree * eventMetaTree = eventTree().metaTree();
    TBranch * auxBranch = eventTree().auxBranch();
    RootTree::EntryNumber entryNumber = eventTree().entryNumber();
    auxBranch->SetAddress(&pEvAux);
    auxBranch->GetEntry(entryNumber);
    eventMetaTree->GetEntry(entryNumber);
    bool isNewRun = (evAux.id().run() != eventAux().id().run() || luminosityBlockPrincipal_.get() == 0);
    bool isNewLumi = isNewRun || (evAux.luminosityBlockID() != eventAux().luminosityBlockID());
    if (isNewRun) {
      boost::shared_ptr<RunPrincipal const> runPrincipal(new RunPrincipal(eventID().run(), productRegistry()));
      luminosityBlockPrincipal_ = boost::shared_ptr<LuminosityBlockPrincipal const>(
		new LuminosityBlockPrincipal(evAux.luminosityBlockID(), productRegistry(), runPrincipal));
    } else if (isNewLumi) {
      boost::shared_ptr<RunPrincipal const> runPrincipal = luminosityBlockPrincipal_->runPrincipalConstSharedPtr();
      luminosityBlockPrincipal_ = boost::shared_ptr<LuminosityBlockPrincipal const>(
		new LuminosityBlockPrincipal(evAux.luminosityBlockID(), productRegistry(), runPrincipal));
    }
    eventAux_ = evAux;
    // We're not done ... so prepare the EventPrincipal
    boost::shared_ptr<DelayedReader> store_(new RootDelayedReader(entryNumber, branches_, filePtr_));
    std::auto_ptr<EventPrincipal> thisEvent(new EventPrincipal(
                eventID(), evAux.time(), pReg,
		luminosityBlockPrincipal_,
		evAux.processHistoryID_, store_));
    // Loop over provenance
    std::vector<BranchEntryDescription>::iterator pit = eventTree().provenance().begin();
    std::vector<BranchEntryDescription>::iterator pitEnd = eventTree().provenance().end();
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
      prov->product = products_[prov->event.productID_];
      bool const isPresent = prov->event.isPresent();
      std::auto_ptr<Group> g(new Group(prov, isPresent));
      // END These lines defer reading branches
      thisEvent->addGroup(g);
    }
    // report event read from file
    Service<JobReport> reportSvc;
    reportSvc->eventReadFromFile(reportToken_, eventID());
    return thisEvent;
  }
}
