/*----------------------------------------------------------------------
$Id: RootFile.cc,v 1.50 2007/01/03 16:08:24 wmtan Exp $
----------------------------------------------------------------------*/

#include "IOPool/Input/src/RootFile.h"

#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/BranchType.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
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
      productRegistry_(new ProductRegistry),
      luminosityBlockPrincipal_() {
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

    assert(fileFormatVersion_.isValid());
    assert(eventTree().isValid());
    if (fileFormatVersion_.value_ >= 2) {
      assert(lumiTree().isValid());
      assert(runTree().isValid());
    }

    // freeze our temporary product registry
    tempReg.setFrozen();

    // Do the translation from the old registry to the new one
    std::map<std::string,std::string> newBranchToOldBranch;
    {
      ProductRegistry::ProductList const& prodList = tempReg.productList();
      for (ProductRegistry::ProductList::const_iterator it = prodList.begin();
           it != prodList.end(); ++it) {
        BranchDescription const& prod = it->second;
	//need to call init to cause the branch name to be recalculated
	prod.init();
        BranchDescription newBD(prod);
        newBD.friendlyClassName_ = friendlyname::friendlyName(newBD.className());

        newBD.init();
        productRegistry_->addProduct(newBD);
	newBranchToOldBranch[newBD.branchName()] = prod.branchName();
      }
      // freeze the product registry
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
      treePointers_[prod.branchType()]->addBranch(it->first, prod,
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
    eventTree().fillAux<EventAux>(pEvAux);
    bool isNewRun = (evAux.id().run() != eventAux().id().run() || luminosityBlockPrincipal_.get() == 0);
    bool isNewLumi = isNewRun || (evAux.luminosityBlockID() != eventAux().luminosityBlockID());
    eventAux_ = evAux;
    if (isNewLumi) {
      luminosityBlockPrincipal_ = readLumi(pReg, evAux.id().run(), evAux.luminosityBlockID(), isNewRun);
    }
    // We're not done ... so prepare the EventPrincipal
    std::auto_ptr<EventPrincipal> thisEvent(new EventPrincipal(
                eventID(), evAux.time(), pReg,
		luminosityBlockPrincipal_, processConfiguration_,
		evAux.processHistoryID_, eventTree().makeDelayedReader()));

    // Create a group in the event for each product
    eventTree().fillGroups(thisEvent->groupGetter());

    // report event read from file
    Service<JobReport> reportSvc;
    reportSvc->eventReadFromFile(reportToken_, eventID().run(), eventID().event());
    return thisEvent;
  }

  boost::shared_ptr<RunPrincipal const>
  RootFile::readRun(ProductRegistry const& pReg, RunNumber_t const& runNumber) {
    if (!runTree().isValid()) {
      return boost::shared_ptr<RunPrincipal const>(new RunPrincipal(runNumber, pReg, processConfiguration_));
    }
    RootTree::EntryNumber entry = runTree().getExactEntryNumber(runNumber, 0);
    if (entry < 0) {
      throw cms::Exception("NotFound", "RootFile::readRun()")
        << "Run " << runNumber << " was not found in file " << file_ << "\n";
    }
    runTree().setEntryNumber(entry);
    RunAux runAux;
    RunAux *pRunAux = &runAux;
    runTree().fillAux<RunAux>(pRunAux);
    assert(runNumber == runAux.id());
    boost::shared_ptr<RunPrincipal> thisRun(new RunPrincipal(runNumber, pReg, processConfiguration_,
		runAux.processHistoryID_, runTree().makeDelayedReader()));
    // Create a group in the run for each product
    runTree().fillGroups(thisRun->groupGetter());
    return thisRun;
  }

  boost::shared_ptr<LuminosityBlockPrincipal const>
  RootFile::readLumi(ProductRegistry const& pReg, RunNumber_t const& runNumber,
						  LuminosityBlockID const& lumiID,
						  bool isNewRun) {
    boost::shared_ptr<RunPrincipal const> runPrincipal = (isNewRun ?
	readRun(pReg, runNumber) :
	luminosityBlockPrincipal_->runPrincipalConstSharedPtr());
    if (!lumiTree().isValid()) {
      return boost::shared_ptr<LuminosityBlockPrincipal const>(
	new LuminosityBlockPrincipal(lumiID, pReg, runPrincipal, processConfiguration_));
    }
    RootTree::EntryNumber entry = lumiTree().getExactEntryNumber(runNumber, lumiID);
    if (entry < 0) {
      throw cms::Exception("NotFound", "RootFile::read()")
        << "Lumi Block " << lumiID << " in Run " << runNumber << " was not found in file " << file_ << "\n";
    }
    lumiTree().setEntryNumber(entry);
    LuminosityBlockAux lumiAux;
    LuminosityBlockAux *pLumiAux = &lumiAux;
    lumiTree().fillAux<LuminosityBlockAux>(pLumiAux);
    assert(lumiID == lumiAux.id());
    assert(runNumber == lumiAux.runID());
    boost::shared_ptr<LuminosityBlockPrincipal> thisLumi(
	new LuminosityBlockPrincipal(lumiID, pReg, runPrincipal, processConfiguration_,
		lumiAux.processHistoryID_, lumiTree().makeDelayedReader()));
    // Create a group in the lumi for each product
    lumiTree().fillGroups(thisLumi->groupGetter());
    return thisLumi;
  }

}
