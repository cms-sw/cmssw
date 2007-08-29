/*----------------------------------------------------------------------
$Id: RootFile.cc,v 1.79 2007/08/10 22:11:26 wmtan Exp $
----------------------------------------------------------------------*/

#include "RootFile.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
//used for friendlyName translation
#include "FWCore/Utilities/interface/FriendlyName.h"

//used for backward compatibility
#include "DataFormats/Provenance/interface/EventAux.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAux.h"
#include "DataFormats/Provenance/interface/RunAux.h"

#include "TFile.h"
#include "TTree.h"
#include "Rtypes.h"

namespace edm {
  namespace {
    void
    kludgeZeroRun(RunID *id) {
      if (id->run() == 0) (*id) = RunID(1);
    }
    void
    kludgeZeroRun(LuminosityBlockID *id) {
      if (id->run() == 0) (*id) = LuminosityBlockID(1, id->luminosityBlock());
    }
    void
    kludgeZeroRun(EventID *id) {
      if (id->run() == 0) (*id) = EventID(1, id->event());
    }
  }
//---------------------------------------------------------------------
  RootFile::RootFile(std::string const& fileName,
		     std::string const& catalogName,
		     ProcessConfiguration const& processConfiguration,
		     std::string const& logicalFileName) :
      file_(fileName),
      logicalFile_(logicalFileName),
      catalog_(catalogName),
      processConfiguration_(processConfiguration),
      filePtr_(file_.empty() ? 0 : TFile::Open(file_.c_str())),
      fileFormatVersion_(),
      reportToken_(0),
      eventAux_(),
      lumiAux_(),
      runAux_(),
      eventTree_(filePtr_, InEvent),
      lumiTree_(filePtr_, InLumi),
      runTree_(filePtr_, InRun),
      treePointers_(),
      productRegistry_() {
    treePointers_[InEvent] = &eventTree_;
    treePointers_[InLumi]  = &lumiTree_;
    treePointers_[InRun]   = &runTree_;

    open();

    // Set up buffers for registries.
    // Need to read to a temporary registry so we can do a translation of the BranchKeys.
    // This preserves backward compatibility against friendly class name algorithm changes.
    ProductRegistry tempReg;
    ProductRegistry *ppReg = &tempReg;
    typedef std::map<ParameterSetID, ParameterSetBlob> PsetMap;
    PsetMap psetMap;
    ProcessHistoryMap pHistMap;
    ModuleDescriptionMap mdMap;
    PsetMap *psetMapPtr = &psetMap;
    ProcessHistoryMap *pHistMapPtr = &pHistMap;
    ModuleDescriptionMap *mdMapPtr = &mdMap;
    FileFormatVersion *fftPtr = &fileFormatVersion_;

    // Read the metadata tree.
    TTree *metaDataTree = dynamic_cast<TTree *>(filePtr_->Get(poolNames::metaDataTreeName().c_str()));
    assert(metaDataTree != 0);

    metaDataTree->SetBranchAddress(poolNames::productDescriptionBranchName().c_str(),(&ppReg));
    metaDataTree->SetBranchAddress(poolNames::parameterSetMapBranchName().c_str(), &psetMapPtr);
    metaDataTree->SetBranchAddress(poolNames::processHistoryMapBranchName().c_str(), &pHistMapPtr);
    metaDataTree->SetBranchAddress(poolNames::moduleDescriptionMapBranchName().c_str(), &mdMapPtr);
    metaDataTree->SetBranchAddress(poolNames::fileFormatVersionBranchName().c_str(), &fftPtr);

    metaDataTree->GetEntry(0);

    validateFile();

    // freeze our temporary product registry
    tempReg.setFrozen();

    ProductRegistry *newReg = new ProductRegistry;
    // Do the translation from the old registry to the new one
    std::map<std::string,std::string> newBranchToOldBranch;
    {
      ProductRegistry::ProductList const& prodList = tempReg.productList();
      for (ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end();
           it != itEnd; ++it) {
        BranchDescription const& prod = it->second;
	//need to call init to cause the branch name to be recalculated
	prod.init();
        BranchDescription newBD(prod);
        newBD.friendlyClassName_ = friendlyname::friendlyName(newBD.className());

        newBD.init();
        newReg->addProduct(newBD);
	newBranchToOldBranch[newBD.branchName()] = prod.branchName();
      }
      // freeze the product registry
      newReg->setFrozen();
      productRegistry_ = boost::shared_ptr<ProductRegistry const>(newReg);
    }

    // Merge into the registries. For now, we do NOT merge the product registry.
    pset::Registry& psetRegistry = *pset::Registry::instance();
    for (PsetMap::const_iterator i = psetMap.begin(), iEnd = psetMap.end(); i != iEnd; ++i) {
      psetRegistry.insertMapped(ParameterSet(i->second.pset_));
    } 
    ProcessHistoryRegistry & processNameListRegistry = *ProcessHistoryRegistry::instance();
    for (ProcessHistoryMap::const_iterator j = pHistMap.begin(), jEnd = pHistMap.end(); j != jEnd; ++j) {
      processNameListRegistry.insertMapped(j->second);
    } 
    ModuleDescriptionRegistry & moduleDescriptionRegistry = *ModuleDescriptionRegistry::instance();
    for (ModuleDescriptionMap::const_iterator k = mdMap.begin(), kEnd = mdMap.end(); k != kEnd; ++k) {
      moduleDescriptionRegistry.insertMapped(k->second);
    } 

    // Set up information from the product registry.
    ProductRegistry::ProductList const& prodList = productRegistry()->productList();
    for (ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end();
        it != itEnd; ++it) {
      BranchDescription const& prod = it->second;
      treePointers_[prod.branchType()]->addBranch(it->first, prod,
						 newBranchToOldBranch[prod.branchName()]);
    }
  }

  RootFile::~RootFile() {
  }

  void RootFile::validateFile() {
    if (!fileFormatVersion_.isValid()) {
      fileFormatVersion_.value_ = 0;
    }
    assert(eventTree().isValid());
//  if (fileFormatVersion_.value_ >= 3) {
//    assert(lumiTree().isValid());
//    assert(runTree().isValid());
//  }
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

  void
  RootFile::fillEventAuxiliary() {
    if (fileFormatVersion_.value_ >= 3) {
      EventAuxiliary *pEvAux = &eventAux_;
      eventTree().fillAux<EventAuxiliary>(pEvAux);
    } else {
      // for backward compatibility.
      EventAux eventAux;
      EventAux *pEvAux = &eventAux;
      eventTree().fillAux<EventAux>(pEvAux);
      conversion(eventAux, eventAux_);
      if (fileFormatVersion_.value_ <= 1) {
        eventAux_.luminosityBlock_ = 1;
      }
    }
    kludgeZeroRun(&eventAux_.id_);
  }

  // readEvent() is responsible for creating, and setting up, the
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
  RootFile::readEvent(boost::shared_ptr<ProductRegistry const> pReg, boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    if (!eventTree().next()) {
      return std::auto_ptr<EventPrincipal>(0);
    }
    fillEventAuxiliary();

    if (lbp.get() == 0) {
	boost::shared_ptr<RunPrincipal> rp(
	  new RunPrincipal(eventAux_.run(), eventAux_.time(), eventAux_.time(), pReg, processConfiguration_));
	lbp = boost::shared_ptr<LuminosityBlockPrincipal>(
	  new LuminosityBlockPrincipal(eventAux_.luminosityBlock(),
				       eventAux_.time(),
				       eventAux_.time(),
				       pReg,
				       rp,
				       processConfiguration_));
    }

    if (eventAux_.run() != lbp->runNumber() ||
	eventAux_.luminosityBlock() != lbp->luminosityBlock()) {
      // The event is in a different run or lumi block.  Back up, and return a null pointer.
      eventTree().previous();
      return std::auto_ptr<EventPrincipal>(0);
    }
    // We're not done ... so prepare the EventPrincipal
    std::auto_ptr<EventPrincipal> thisEvent(new EventPrincipal(
                eventID(),
		eventAux_.time(), pReg,
		lbp, processConfiguration_,
		eventAux_.isRealData(),
		eventAux_.experimentType(),
		eventAux_.bunchCrossing(),
                eventAux_.storeNumber(),
		eventAux_.processHistoryID_,
		eventTree().makeDelayedReader()));

    // Create a group in the event for each product
    eventTree().fillGroups(thisEvent->groupGetter());

    // report event read from file
    Service<JobReport> reportSvc;
    reportSvc->eventReadFromFile(reportToken_, eventID().run(), eventID().event());
    return thisEvent;
  }

  boost::shared_ptr<RunPrincipal>
  RootFile::readRun(boost::shared_ptr<ProductRegistry const> pReg) {
    if (!runTree().isValid()) {
      // prior to the support of run trees, the run number must be retrieved from the next event.
      if (!eventTree().next()) {
        return boost::shared_ptr<RunPrincipal>();
      }
      EventAux eventAux;
      EventAux *pEvAux = &eventAux;
      eventTree().fillAux<EventAux>(pEvAux);
      kludgeZeroRun(&eventAux.id_);
      // back up, so event will not be skipped.
      eventTree().previous();
      return boost::shared_ptr<RunPrincipal>(
          new RunPrincipal(eventAux.id_.run(),
	  eventAux.time_,
	  Timestamp::invalidTimestamp(), pReg,
	  processConfiguration_));
    }
    if (!runTree().next()) {
      return boost::shared_ptr<RunPrincipal>();
    }
    if (fileFormatVersion_.value_ >= 3) {
      RunAuxiliary *pRunAux = &runAux_;
      runTree().fillAux<RunAuxiliary>(pRunAux);
    } else {
      RunAux runAux;
      RunAux *pRunAux = &runAux;
      runTree().fillAux<RunAux>(pRunAux);
      conversion(runAux, runAux_);
    } 
    kludgeZeroRun(&runAux_.id_);
    if (runAux_.beginTime() == Timestamp::invalidTimestamp()) {
      // RunAuxiliary did not contain a valid timestamp.  Take it from the next event.
      if (eventTree().next()) {
        fillEventAuxiliary();
        // back up, so event will not be skipped.
        eventTree().previous();
      }
      runAux_.beginTime_ = eventAux_.time(); 
      runAux_.endTime_ = Timestamp::invalidTimestamp();
    }
    boost::shared_ptr<RunPrincipal> thisRun(
	new RunPrincipal(runAux_.run(),
			 runAux_.beginTime(),
			 runAux_.endTime(),
			 pReg,
			 processConfiguration_,
			 runAux_.processHistoryID_,
			 runTree().makeDelayedReader()));
    // Create a group in the run for each product
    runTree().fillGroups(thisRun->groupGetter());
    return thisRun;
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  RootFile::readLumi(boost::shared_ptr<ProductRegistry const> pReg, boost::shared_ptr<RunPrincipal> rp) {
    if (!lumiTree().isValid()) {
      // prior to the support of lumi trees, the run number must be retrieved from the next event.
      if (!eventTree().next()) {
        return boost::shared_ptr<LuminosityBlockPrincipal>();
      }
      EventAux eventAux;
      EventAux *pEvAux = &eventAux;
      eventTree().fillAux<EventAux>(pEvAux);
      kludgeZeroRun(&eventAux.id_);
      // back up, so event will not be skipped.
      eventTree().previous();
      if (eventAux.id_.run() != rp->run()) {
        // The next event is in a different run.  Return a null pointer.
        return boost::shared_ptr<LuminosityBlockPrincipal>();
      }
      // Prior to support of lumi blocks, always use 1 for lumi block number.
      return boost::shared_ptr<LuminosityBlockPrincipal>(
	new LuminosityBlockPrincipal(1,
				     eventAux.time_,
				     Timestamp::invalidTimestamp(),
				     pReg,
				     rp,
				     processConfiguration_));
    }
    if (!lumiTree().next()) {
      return boost::shared_ptr<LuminosityBlockPrincipal>();
    }
    if (fileFormatVersion_.value_ >= 3) {
      LuminosityBlockAuxiliary *pLumiAux = &lumiAux_;
      lumiTree().fillAux<LuminosityBlockAuxiliary>(pLumiAux);
    } else {
      LuminosityBlockAux lumiAux;
      LuminosityBlockAux *pLumiAux = &lumiAux;
      lumiTree().fillAux<LuminosityBlockAux>(pLumiAux);
      conversion(lumiAux, lumiAux_);
    }
    kludgeZeroRun(&lumiAux_.id_);

    if (lumiAux_.run() != rp->run()) {
      // The lumi block is in a different run.  Back up, and return a null pointer.
      lumiTree().previous();
      return boost::shared_ptr<LuminosityBlockPrincipal>();
    }
    if (lumiAux_.beginTime() == Timestamp::invalidTimestamp()) {
      // LuminosityBlockAuxiliary did not contain a timestamp. Take it from the next event.
      if (eventTree().next()) {
        fillEventAuxiliary();
        // back up, so event will not be skipped.
        eventTree().previous();
      }
      lumiAux_.beginTime_ = eventAux_.time();
      lumiAux_.endTime_ = Timestamp::invalidTimestamp();
    }
    boost::shared_ptr<LuminosityBlockPrincipal> thisLumi(
	new LuminosityBlockPrincipal(lumiAux_.luminosityBlock(),
				     lumiAux_.beginTime(),
				     lumiAux_.endTime(),
				     pReg, rp, processConfiguration_,
				     lumiAux_.processHistoryID_,
				     lumiTree().makeDelayedReader()));
    // Create a group in the lumi for each product
    lumiTree().fillGroups(thisLumi->groupGetter());
    return thisLumi;
  }
}
