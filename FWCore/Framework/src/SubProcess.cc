#include "FWCore/Framework/interface/SubProcess.h"

#include "DataFormats/Common/interface/ProductData.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/ProductHolder.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/OutputModuleDescription.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/getAllTriggerNames.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/src/EventSetupsController.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/ParameterSet/interface/IllegalParameters.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ExceptionCollector.h"

#include <cassert>
#include <string>
#include <vector>

namespace edm {

  SubProcess::SubProcess(ParameterSet& parameterSet,
                         ParameterSet const& topLevelParameterSet,
                         boost::shared_ptr<ProductRegistry const> parentProductRegistry,
                         boost::shared_ptr<BranchIDListHelper const> parentBranchIDListHelper,
                         eventsetup::EventSetupsController& esController,
                         ActivityRegistry& parentActReg,
                         ServiceToken const& token,
                         serviceregistry::ServiceLegacy iLegacy,
                         PreallocationConfiguration const& preallocConfig,
                         ProcessContext const* parentProcessContext) :
      serviceToken_(),
      parentPreg_(parentProductRegistry),
      preg_(),
      branchIDListHelper_(),
      act_table_(),
      processConfiguration_(),
      historyLumiOffset_(preallocConfig.numberOfStreams()),
      historyRunOffset_(historyLumiOffset_+preallocConfig.numberOfLuminosityBlocks()),
      processHistoryRegistries_(historyRunOffset_+ preallocConfig.numberOfRuns()),
      historyAppenders_(historyRunOffset_+preallocConfig.numberOfRuns()),
      principalCache_(),
      esp_(),
      schedule_(),
      parentToChildPhID_(),
      subProcess_(),
      processParameterSet_(),
      productSelectorRules_(parameterSet, "outputCommands", "OutputModule"),
      productSelector_(),
      wantAllEvents_(true) {
  
    //Setup the event selection
    Service<service::TriggerNamesService> tns;
    
    ParameterSet selectevents =
    parameterSet.getUntrackedParameterSet("SelectEvents", ParameterSet());
    
    selectevents.registerIt(); // Just in case this PSet is not registered
    wantAllEvents_ = detail::configureEventSelector(selectevents,
                                                    tns->getProcessName(),
                                                    getAllTriggerNames(),
                                                    selectors_);
    std::map<std::string, std::vector<std::pair<std::string, int> > > outputModulePathPositions;
    selector_config_id_ = detail::registerProperSelectionInfo(selectevents,
                                                              "",
                                                              outputModulePathPositions,
                                                              parentProductRegistry->anyProductProduced());
        
    selectProducts(*parentProductRegistry);

    std::string const maxEvents("maxEvents");
    std::string const maxLumis("maxLuminosityBlocks");

    processParameterSet_.reset(parameterSet.popParameterSet(std::string("process")).release()); 

    // if this process has a maxEvents or maxLuminosityBlocks parameter set, remove them.
    if(processParameterSet_->exists(maxEvents)) {
      processParameterSet_->popParameterSet(maxEvents);
    }
    if(processParameterSet_->exists(maxLumis)) {
      processParameterSet_->popParameterSet(maxLumis);
    }

    // if the top level process has a maxEvents or maxLuminosityBlocks parameter set, add them to this process.
    if(topLevelParameterSet.exists(maxEvents)) {
      processParameterSet_->addUntrackedParameter<ParameterSet>(maxEvents, topLevelParameterSet.getUntrackedParameterSet(maxEvents));
    }
    if(topLevelParameterSet.exists(maxLumis)) {
      processParameterSet_->addUntrackedParameter<ParameterSet>(maxLumis, topLevelParameterSet.getUntrackedParameterSet(maxLumis));
    }

    // If this process has a subprocess, pop the subprocess parameter set out of the process parameter set

    boost::shared_ptr<ParameterSet> subProcessParameterSet(popSubProcessParameterSet(*processParameterSet_).release());
  
    ScheduleItems items(*parentProductRegistry, *parentBranchIDListHelper, *this);

    ParameterSet const& optionsPset(processParameterSet_->getUntrackedParameterSet("options", ParameterSet()));
    IllegalParameters::setThrowAnException(optionsPset.getUntrackedParameter<bool>("throwIfIllegalParameter", true));

    //initialize the services
    ServiceToken iToken;

    // get any configured services.
    std::auto_ptr<std::vector<ParameterSet> > serviceSets = processParameterSet_->popVParameterSet(std::string("services")); 

    ServiceToken newToken = items.initServices(*serviceSets, *processParameterSet_, token, iLegacy, false);
    parentActReg.connectToSubProcess(*items.actReg_);
    serviceToken_ = items.addCPRandTNS(*processParameterSet_, newToken);


    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    // intialize miscellaneous items
    items.initMisc(*processParameterSet_);

    // intialize the event setup provider
    esp_ = esController.makeProvider(*processParameterSet_);

    // intialize the Schedule
    schedule_ = items.initSchedule(*processParameterSet_,subProcessParameterSet.get(),preallocConfig,&processContext_);

    // set the items
    act_table_ = std::move(items.act_table_);
    preg_.reset(items.preg_.release());
    //CMS-THREADING this only works since Run/Lumis are synchronous so when principalCache asks for
    // the reducedProcessHistoryID from a full ProcessHistoryID that registry will not be in use by
    // another thread. We really need to change how this is done in the PrincipalCache.
    principalCache_.setProcessHistoryRegistry(processHistoryRegistries_[historyRunOffset_]);
    branchIDListHelper_ = items.branchIDListHelper_;
    processConfiguration_ = items.processConfiguration_;
    processContext_.setProcessConfiguration(processConfiguration_.get());
    processContext_.setParentProcessContext(parentProcessContext);

    principalCache_.setNumberOfConcurrentPrincipals(preallocConfig);
    for(unsigned int index = 0; index < preallocConfig.numberOfStreams(); ++index) {
      boost::shared_ptr<EventPrincipal> ep(new EventPrincipal(preg_,
                                                              branchIDListHelper_,
                                                              *processConfiguration_,
                                                              &(historyAppenders_[index]),
                                                              index));
      ep->preModuleDelayedGetSignal_.connect(std::cref(items.actReg_->preModuleEventDelayedGetSignal_));
      ep->postModuleDelayedGetSignal_.connect(std::cref(items.actReg_->postModuleEventDelayedGetSignal_));
      principalCache_.insert(ep);
    }
    if(subProcessParameterSet) {
      subProcess_.reset(new SubProcess(*subProcessParameterSet,
                                       topLevelParameterSet,
                                       preg_,
                                       branchIDListHelper_,
                                       esController,
                                       *items.actReg_,
                                       newToken,
                                       iLegacy,
                                       preallocConfig,
                                       &processContext_));
    }
  }

  SubProcess::~SubProcess() {}

  void
  SubProcess::doBeginJob() {
    this->beginJob();
  }
  
  void
  SubProcess::doEndJob() {
    endJob();
  }
  

  void
  SubProcess::beginJob() {
    if(!droppedBranchIDToKeptBranchID().empty()) {
      fixBranchIDListsForEDAliases(droppedBranchIDToKeptBranchID());
    }
    ServiceRegistry::Operate operate(serviceToken_);
    schedule_->beginJob(*preg_);
    if(subProcess_.get()) subProcess_->doBeginJob();
  }

  void
  SubProcess::endJob() {
    ServiceRegistry::Operate operate(serviceToken_);
    ExceptionCollector c("Multiple exceptions were thrown while executing endJob. An exception message follows for each.");
    schedule_->endJob(c);
    if(subProcess_.get()) c.call([this](){ this->subProcess_->doEndJob();});
    if(c.hasThrown()) {
      c.rethrow();
    }
  }

  void
  SubProcess::selectProducts(ProductRegistry const& preg) {
    if(productSelector_.initialized()) return;
    productSelector_.initialize(productSelectorRules_, preg.allBranchDescriptions());
    
    // TODO: See if we can collapse keptProducts_ and productSelector_ into a
    // single object. See the notes in the header for ProductSelector
    // for more information.
    
    std::map<BranchID, BranchDescription const*> trueBranchIDToKeptBranchDesc;
    
    for(auto const& it : preg.productList()) {
      BranchDescription const& desc = it.second;
      if(desc.transient()) {
        // if the class of the branch is marked transient, output nothing
      } else if(!desc.present() && !desc.produced()) {
        // else if the branch containing the product has been previously dropped,
        // output nothing
      } else if(productSelector_.selected(desc)) {
        // else if the branch has been selected, put it in the list of selected branches.
        if(desc.produced()) {
          // First we check if an equivalent branch has already been selected due to an EDAlias.
          // We only need the check for products produced in this process.
          BranchID const& trueBranchID = desc.originalBranchID();
          std::map<BranchID, BranchDescription const*>::const_iterator iter = trueBranchIDToKeptBranchDesc.find(trueBranchID);
          if(iter != trueBranchIDToKeptBranchDesc.end()) {
            throw edm::Exception(errors::Configuration, "Duplicate Output Selection")
            << "Two (or more) equivalent branches have been selected for output.\n"
            << "#1: " << BranchKey(desc) << "\n"
            << "#2: " << BranchKey(*iter->second) << "\n"
            << "Please drop at least one of them.\n";
          }
          trueBranchIDToKeptBranchDesc.insert(std::make_pair(trueBranchID, &desc));
        }
        // Now put it in the list of selected branches.
        keptProducts_[desc.branchType()].push_back(&desc);
      }
    }
    // Now fill in a mapping needed in the case that a branch was dropped while its EDAlias was kept.
    for(auto const& it : preg.productList()) {
      BranchDescription const& desc = it.second;
      if(!desc.produced() || desc.isAlias()) continue;
      BranchID const& branchID = desc.branchID();
      std::map<BranchID, BranchDescription const*>::const_iterator iter = trueBranchIDToKeptBranchDesc.find(branchID);
      if(iter != trueBranchIDToKeptBranchDesc.end()) {
        // This branch, produced in this process, or an alias of it, was persisted.
        BranchID const& keptBranchID = iter->second->branchID();
        if(keptBranchID != branchID) {
          // An EDAlias branch was persisted.
          droppedBranchIDToKeptBranchID_.insert(std::make_pair(branchID.id(), keptBranchID.id()));
        }
      }
    }
  }

  void
  SubProcess::fixBranchIDListsForEDAliases(std::map<BranchID::value_type, BranchID::value_type> const& droppedBranchIDToKeptBranchID) {
    // Check for branches dropped while an EDAlias was kept.
    // Replace BranchID of each dropped branch with that of the kept alias.
    for(BranchIDList& branchIDList : branchIDListHelper_->mutableBranchIDLists()) {
      for(BranchID::value_type& branchID : branchIDList) {
        std::map<BranchID::value_type, BranchID::value_type>::const_iterator iter = droppedBranchIDToKeptBranchID.find(branchID);
        if(iter != droppedBranchIDToKeptBranchID.end()) {
          branchID = iter->second;
        }
      }
    }
    if(subProcess_.get()) subProcess_->fixBranchIDListsForEDAliases(droppedBranchIDToKeptBranchID);
  }

  void
  SubProcess::doEvent(EventPrincipal const& ep) {
    ServiceRegistry::Operate operate(serviceToken_);
    /* BEGIN relevant bits from OutputModule::doEvent */
    detail::TRBESSentry products_sentry(selectors_);
    
    
    if(!wantAllEvents_) {
      // use module description and const_cast unless interface to
      // event is changed to just take a const EventPrincipal
      if(!selectors_.wantEvent(ep, nullptr)) {
        return;
      }
    }
    process(ep);
    /* END relevant bits from OutputModule::doEvent */
  }

  void
  SubProcess::process(EventPrincipal const& principal) {
    EventAuxiliary aux(principal.aux());
    aux.setProcessHistoryID(principal.processHistoryID());

    EventSelectionIDVector esids{principal.eventSelectionIDs()};
    if (principal.productRegistry().anyProductProduced() || !wantAllEvents_) {
      esids.push_back(selector_config_id_);
    }

    EventPrincipal& ep = principalCache_.eventPrincipal(principal.streamID().value());
    auto & processHistoryRegistry = processHistoryRegistries_[principal.streamID().value()];
    processHistoryRegistry.registerProcessHistory(principal.processHistory());
    BranchListIndexes bli(principal.branchListIndexes());
    ep.fillEventPrincipal(aux,
                          processHistoryRegistry,
                          std::move(esids),
                          std::move(bli),
                          *(principal.productProvenanceRetrieverPtr()),//NOTE: this transfers the per product provenance
                          principal.reader());
    ep.setLuminosityBlockPrincipal(principalCache_.lumiPrincipalPtr());
    propagateProducts(InEvent, principal, ep);
    typedef OccurrenceTraits<EventPrincipal, BranchActionStreamBegin> Traits;
    schedule_->processOneEvent<Traits>(ep.streamID().value(),ep, esp_->eventSetup());
    if(subProcess_.get()) subProcess_->doEvent(ep);
    ep.clearEventPrincipal();
  }

  void
  SubProcess::doBeginRun(RunPrincipal const& principal, IOVSyncValue const& ts) {
    ServiceRegistry::Operate operate(serviceToken_);
    beginRun(principal,ts);
  }

  void
  SubProcess::beginRun(RunPrincipal const& principal, IOVSyncValue const& ts) {
    boost::shared_ptr<RunAuxiliary> aux(new RunAuxiliary(principal.aux()));
    aux->setProcessHistoryID(principal.processHistoryID());
    boost::shared_ptr<RunPrincipal> rpp(new RunPrincipal(aux, preg_, *processConfiguration_, &(historyAppenders_[historyRunOffset_+principal.index()]),principal.index()));
    auto & processHistoryRegistry = processHistoryRegistries_[historyRunOffset_+principal.index()];
    processHistoryRegistry.registerProcessHistory(principal.processHistory());
    rpp->fillRunPrincipal(processHistoryRegistry, principal.reader());
    principalCache_.insert(rpp);

    ProcessHistoryID const& parentInputReducedPHID = principal.reducedProcessHistoryID();
    ProcessHistoryID const& inputReducedPHID       = rpp->reducedProcessHistoryID();

    parentToChildPhID_.insert(std::make_pair(parentInputReducedPHID,inputReducedPHID));

    RunPrincipal& rp = *principalCache_.runPrincipalPtr();
    propagateProducts(InRun, principal, rp);
    typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin> Traits;
    schedule_->processOneGlobal<Traits>(rp, esp_->eventSetupForInstance(ts));
    if(subProcess_.get()) subProcess_->doBeginRun(rp, ts);
  }

  void
  SubProcess::doEndRun(RunPrincipal const& principal, IOVSyncValue const& ts, bool cleaningUpAfterException) {
    ServiceRegistry::Operate operate(serviceToken_);
    endRun(principal,ts,cleaningUpAfterException);
  }

  void
  SubProcess::endRun(RunPrincipal const& principal, IOVSyncValue const& ts, bool cleaningUpAfterException) {
    RunPrincipal& rp = *principalCache_.runPrincipalPtr();
    propagateProducts(InRun, principal, rp);
    typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd> Traits;
    schedule_->processOneGlobal<Traits>(rp, esp_->eventSetupForInstance(ts), cleaningUpAfterException);
    if(subProcess_.get()) subProcess_->doEndRun(rp, ts, cleaningUpAfterException);
  }

  void
  SubProcess::writeRun(ProcessHistoryID const& parentPhID, int runNumber) {
    ServiceRegistry::Operate operate(serviceToken_);
    std::map<ProcessHistoryID, ProcessHistoryID>::const_iterator it = parentToChildPhID_.find(parentPhID);
    assert(it != parentToChildPhID_.end());
    schedule_->writeRun(principalCache_.runPrincipal(it->second, runNumber), &processContext_);
    if(subProcess_.get()) subProcess_->writeRun(it->second, runNumber);
  }

  void
  SubProcess::deleteRunFromCache(ProcessHistoryID const& parentPhID, int runNumber) {
    std::map<ProcessHistoryID, ProcessHistoryID>::const_iterator it = parentToChildPhID_.find(parentPhID);
    assert(it != parentToChildPhID_.end());
    principalCache_.deleteRun(it->second, runNumber);
    if(subProcess_.get()) subProcess_->deleteRunFromCache(it->second, runNumber);
  }

  void
  SubProcess::doBeginLuminosityBlock(LuminosityBlockPrincipal const& principal, IOVSyncValue const& ts) {
    ServiceRegistry::Operate operate(serviceToken_);
    beginLuminosityBlock(principal,ts);
  }

  void
  SubProcess::beginLuminosityBlock(LuminosityBlockPrincipal const& principal, IOVSyncValue const& ts) {
    boost::shared_ptr<LuminosityBlockAuxiliary> aux(new LuminosityBlockAuxiliary(principal.aux()));
    aux->setProcessHistoryID(principal.processHistoryID());
    boost::shared_ptr<LuminosityBlockPrincipal> lbpp(new LuminosityBlockPrincipal(aux, preg_, *processConfiguration_, &(historyAppenders_[historyLumiOffset_+principal.index()]),principal.index()));
    auto & processHistoryRegistry = processHistoryRegistries_[historyLumiOffset_+principal.index()];
    processHistoryRegistry.registerProcessHistory(principal.processHistory());
    lbpp->fillLuminosityBlockPrincipal(processHistoryRegistry, principal.reader());
    lbpp->setRunPrincipal(principalCache_.runPrincipalPtr());
    principalCache_.insert(lbpp);
    LuminosityBlockPrincipal& lbp = *principalCache_.lumiPrincipalPtr();
    propagateProducts(InLumi, principal, lbp);
    typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> Traits;
    schedule_->processOneGlobal<Traits>(lbp, esp_->eventSetupForInstance(ts));
    if(subProcess_.get()) subProcess_->doBeginLuminosityBlock(lbp, ts);
  }

  void
  SubProcess::doEndLuminosityBlock(LuminosityBlockPrincipal const& principal, IOVSyncValue const& ts, bool cleaningUpAfterException) {
    ServiceRegistry::Operate operate(serviceToken_);
    endLuminosityBlock(principal,ts,cleaningUpAfterException);
  }

  void
  SubProcess::endLuminosityBlock(LuminosityBlockPrincipal const& principal, IOVSyncValue const& ts, bool cleaningUpAfterException) {
    LuminosityBlockPrincipal& lbp = *principalCache_.lumiPrincipalPtr();
    propagateProducts(InLumi, principal, lbp);
    typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> Traits;
    schedule_->processOneGlobal<Traits>(lbp, esp_->eventSetupForInstance(ts), cleaningUpAfterException);
    if(subProcess_.get()) subProcess_->doEndLuminosityBlock(lbp, ts, cleaningUpAfterException);
  }

  void
  SubProcess::writeLumi(ProcessHistoryID const& parentPhID, int runNumber, int lumiNumber) {
    ServiceRegistry::Operate operate(serviceToken_);
    std::map<ProcessHistoryID, ProcessHistoryID>::const_iterator it = parentToChildPhID_.find(parentPhID);
    assert(it != parentToChildPhID_.end());
    schedule_->writeLumi(principalCache_.lumiPrincipal(it->second, runNumber, lumiNumber), &processContext_);
    if(subProcess_.get()) subProcess_->writeLumi(it->second, runNumber, lumiNumber);
  }

  void
  SubProcess::deleteLumiFromCache(ProcessHistoryID const& parentPhID, int runNumber, int lumiNumber) {
    std::map<ProcessHistoryID, ProcessHistoryID>::const_iterator it = parentToChildPhID_.find(parentPhID);
    assert(it != parentToChildPhID_.end());
    principalCache_.deleteLumi(it->second, runNumber, lumiNumber);
      if(subProcess_.get()) subProcess_->deleteLumiFromCache(it->second, runNumber, lumiNumber);
  }
  
  void
  SubProcess::doBeginStream(unsigned int iID) {
    ServiceRegistry::Operate operate(serviceToken_);
    schedule_->beginStream(iID);
    if(subProcess_.get()) subProcess_->doBeginStream(iID);
  }

  void
  SubProcess::doEndStream(unsigned int iID) {
    ServiceRegistry::Operate operate(serviceToken_);
    schedule_->endStream(iID);
    if(subProcess_.get()) subProcess_->doEndStream(iID);
  }

  void
  SubProcess::doStreamBeginRun(unsigned int id, RunPrincipal const& principal, IOVSyncValue const& ts) {
    ServiceRegistry::Operate operate(serviceToken_);
    {
      RunPrincipal& rp = *principalCache_.runPrincipalPtr();
      typedef OccurrenceTraits<RunPrincipal, BranchActionStreamBegin> Traits;
      schedule_->processOneStream<Traits>(id,rp, esp_->eventSetupForInstance(ts));
      if(subProcess_.get()) subProcess_->doStreamBeginRun(id,rp, ts);
    }
  }
  
  void
  SubProcess::doStreamEndRun(unsigned int id, RunPrincipal const& principal, IOVSyncValue const& ts, bool cleaningUpAfterException) {
    ServiceRegistry::Operate operate(serviceToken_);
    {
      RunPrincipal& rp = *principalCache_.runPrincipalPtr();
      typedef OccurrenceTraits<RunPrincipal, BranchActionStreamEnd> Traits;
      schedule_->processOneStream<Traits>(id,rp, esp_->eventSetupForInstance(ts),cleaningUpAfterException);
      if(subProcess_.get()) subProcess_->doStreamEndRun(id,rp, ts,cleaningUpAfterException);
    }
  }
  
  void
  SubProcess::doStreamBeginLuminosityBlock(unsigned int id, LuminosityBlockPrincipal const& principal, IOVSyncValue const& ts) {
    ServiceRegistry::Operate operate(serviceToken_);
    {
      LuminosityBlockPrincipal& lbp = *principalCache_.lumiPrincipalPtr();
      typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamBegin> Traits;
      schedule_->processOneStream<Traits>(id,lbp, esp_->eventSetupForInstance(ts));
      if(subProcess_.get()) subProcess_->doStreamBeginLuminosityBlock(id,lbp, ts);
    }
  }
  
  void
  SubProcess::doStreamEndLuminosityBlock(unsigned int id, LuminosityBlockPrincipal const& principal, IOVSyncValue const& ts, bool cleaningUpAfterException) {
    ServiceRegistry::Operate operate(serviceToken_);
    {
      LuminosityBlockPrincipal& lbp = *principalCache_.lumiPrincipalPtr();
      typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionStreamEnd> Traits;
      schedule_->processOneStream<Traits>(id,lbp, esp_->eventSetupForInstance(ts),cleaningUpAfterException);
      if(subProcess_.get()) subProcess_->doStreamEndLuminosityBlock(id,lbp, ts,cleaningUpAfterException);
    }
  }


  void
  SubProcess::propagateProducts(BranchType type, Principal const& parentPrincipal, Principal& principal) const {
    SelectedProducts const& keptVector = keptProducts()[type];
    for(auto const& item : keptVector) {
      ProductHolderBase const* parentProductHolder = parentPrincipal.getProductHolder(item->branchID());
      if(parentProductHolder != nullptr) {
        ProductData const& parentData = parentProductHolder->productData();
        ProductHolderBase const* productHolder = principal.getProductHolder(item->branchID());
        if(productHolder != nullptr) {
          ProductData& thisData = const_cast<ProductData&>(productHolder->productData());
          //Propagate the per event(run)(lumi) data for this product to the subprocess.
          //First, the product itself.
          thisData.wrapper_ = parentData.wrapper_;
          // Then the product ID and the ProcessHistory 
          thisData.prov_.setProductID(parentData.prov_.productID());
          thisData.prov_.setProcessHistory(parentData.prov_.processHistory());
          // Then the store, in case the product needs reading in a subprocess.
          thisData.prov_.setStore(parentData.prov_.store());
          // And last, the other per event provenance.
          if(parentData.prov_.productProvenanceValid()) {
            thisData.prov_.setProductProvenance(*parentData.prov_.productProvenance());
          } else {
            thisData.prov_.resetProductProvenance();
          }
          // Sets unavailable flag, if known that product is not available
          (void)productHolder->productUnavailable();
        }
      }
    }
  }

  // Call respondToOpenInputFile() on all Modules
  void
  SubProcess::respondToOpenInputFile(FileBlock const& fb) {
    ServiceRegistry::Operate operate(serviceToken_);
    branchIDListHelper_->updateFromInput(fb.branchIDLists());
    schedule_->respondToOpenInputFile(fb);
    if(subProcess_.get()) subProcess_->respondToOpenInputFile(fb);
  }

  // free function
  std::auto_ptr<ParameterSet>
  popSubProcessParameterSet(ParameterSet& parameterSet) {
    std::vector<std::string> subProcesses = parameterSet.getUntrackedParameter<std::vector<std::string> >("@all_subprocesses");
    if(!subProcesses.empty()) {
      assert(subProcesses.size() == 1U);
      assert(subProcesses[0] == "@sub_process");
      return parameterSet.popParameterSet(subProcesses[0]);
    }
    return std::auto_ptr<ParameterSet>(nullptr);
  }
}

