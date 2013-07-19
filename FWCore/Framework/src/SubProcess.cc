#include "FWCore/Framework/interface/SubProcess.h"

#include "DataFormats/Common/interface/ProductData.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/FullHistoryToReducedHistoryMap.h"
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
#include "FWCore/Framework/src/EventSetupsController.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
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
                         serviceregistry::ServiceLegacy iLegacy) :
      OutputModule(parameterSet),
      serviceToken_(),
      parentPreg_(parentProductRegistry),
      preg_(),
      branchIDListHelper_(),
      act_table_(),
      processConfiguration_(),
      principalCache_(),
      esp_(),
      schedule_(),
      parentToChildPhID_(),
      historyAppender_(new HistoryAppender),
      esInfo_(nullptr),
      subProcess_(),
      cleaningUpAfterException_(false),
      processParameterSet_() {

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
    schedule_ = items.initSchedule(*processParameterSet_,subProcessParameterSet.get(),StreamID{0});

    // set the items
    act_table_ = std::move(items.act_table_);
    preg_.reset(items.preg_.release());
    branchIDListHelper_ = items.branchIDListHelper_;
    processConfiguration_ = items.processConfiguration_;

    OutputModuleDescription desc(branchIDListHelper_->branchIDLists());
    configure(desc);

    std::map<std::string, std::vector<std::pair<std::string, int> > > outputModulePathPositions;
    setEventSelectionInfo(outputModulePathPositions, parentProductRegistry->anyProductProduced());

    boost::shared_ptr<EventPrincipal> ep(new EventPrincipal(preg_, branchIDListHelper_, *processConfiguration_, historyAppender_.get(),
                                                            StreamID::invalidStreamID()));
    principalCache_.insert(ep);

    if(subProcessParameterSet) {
      subProcess_.reset(new SubProcess(*subProcessParameterSet, topLevelParameterSet, preg_, branchIDListHelper_, esController, *items.actReg_, newToken, iLegacy));
    }
  }

  SubProcess::~SubProcess() {}

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
  SubProcess::fixBranchIDListsForEDAliases(std::map<BranchID::value_type, BranchID::value_type> const& droppedBranchIDToKeptBranchID) {
    // Check for branches dropped while an EDAlias was kept.
    // Replace BranchID of each dropped branch with that of the kept alias.
    for(BranchIDList& branchIDList : branchIDListHelper_->branchIDLists()) {
      for(BranchID::value_type& branchID : branchIDList) {
        std::map<BranchID::value_type, BranchID::value_type>::const_iterator iter = droppedBranchIDToKeptBranchID.find(branchID);
        if(iter != droppedBranchIDToKeptBranchID.end()) {
          branchID = iter->second;
        }
      }
    }
    if(subProcess_.get()) subProcess_->fixBranchIDListsForEDAliases(droppedBranchIDToKeptBranchID);
  }

  SubProcess::ESInfo::ESInfo(IOVSyncValue const& ts, eventsetup::EventSetupProvider& esp) :
      ts_(ts),
      es_(esp.eventSetupForInstance(ts)) {
  }

  void
  SubProcess::doEvent(EventPrincipal const& principal, IOVSyncValue const& ts) {
    ServiceRegistry::Operate operate(serviceToken_);
    esInfo_.reset(new ESInfo(ts, *esp_));
    CurrentProcessingContext cpc;
    doEvent(principal, esInfo_->es_, &cpc);
    esInfo_.reset();
  }

  void
  SubProcess::write(EventPrincipal const& principal) {
    EventAuxiliary aux(principal.aux());
    aux.setProcessHistoryID(principal.processHistoryID());

    boost::shared_ptr<EventSelectionIDVector> esids(new EventSelectionIDVector);
    *esids = principal.eventSelectionIDs();
    if (principal.productRegistry().anyProductProduced() || !wantAllEvents()) {
      esids->push_back(selectorConfig());
    }

    EventPrincipal& ep = principalCache_.eventPrincipal();
    ep.setStreamID(principal.streamID());
    ep.fillEventPrincipal(aux,
                          esids,
                          boost::shared_ptr<BranchListIndexes>(new BranchListIndexes(principal.branchListIndexes())),
                          principal.branchMapperPtr(),
                          principal.reader());
    ep.setLuminosityBlockPrincipal(principalCache_.lumiPrincipalPtr());
    propagateProducts(InEvent, principal, ep);
    typedef OccurrenceTraits<EventPrincipal, BranchActionStreamBegin> Traits;
    schedule_->processOneOccurrence<Traits>(ep, esInfo_->es_);
    if(subProcess_.get()) subProcess_->doEvent(ep, esInfo_->ts_);
    ep.clearEventPrincipal();
  }

  void
  SubProcess::doBeginRun(RunPrincipal const& principal, IOVSyncValue const& ts) {
    ServiceRegistry::Operate operate(serviceToken_);
    esInfo_.reset(new ESInfo(ts, *esp_));
    CurrentProcessingContext cpc;
    doBeginRun(principal, esInfo_->es_, &cpc);
    esInfo_.reset();
  }

  void
  SubProcess::beginRun(RunPrincipal const& principal) {
    boost::shared_ptr<RunAuxiliary> aux(new RunAuxiliary(principal.aux()));
    aux->setProcessHistoryID(principal.processHistoryID());
    boost::shared_ptr<RunPrincipal> rpp(new RunPrincipal(aux, preg_, *processConfiguration_, historyAppender_.get(),principal.index()));
    rpp->fillRunPrincipal(principal.reader());
    principalCache_.insert(rpp);

    FullHistoryToReducedHistoryMap & phidConverter(ProcessHistoryRegistry::instance()->extra());
    ProcessHistoryID const& parentInputReducedPHID = phidConverter.reduceProcessHistoryID(principal.aux().processHistoryID());
    ProcessHistoryID const& inputReducedPHID       = phidConverter.reduceProcessHistoryID(principal.processHistoryID());

    parentToChildPhID_.insert(std::make_pair(parentInputReducedPHID,inputReducedPHID));

    RunPrincipal& rp = *principalCache_.runPrincipalPtr();
    propagateProducts(InRun, principal, rp);
    typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalBegin> Traits;
    schedule_->processOneOccurrence<Traits>(rp, esInfo_->es_);
    if(subProcess_.get()) subProcess_->doBeginRun(rp, esInfo_->ts_);
  }

  void
  SubProcess::doEndRun(RunPrincipal const& principal, IOVSyncValue const& ts, bool cleaningUpAfterException) {
    cleaningUpAfterException_ = cleaningUpAfterException;
    ServiceRegistry::Operate operate(serviceToken_);
    esInfo_.reset(new ESInfo(ts, *esp_));
    CurrentProcessingContext cpc;
    doEndRun(principal, esInfo_->es_, &cpc);
    esInfo_.reset();
  }

  void
  SubProcess::endRun(RunPrincipal const& principal) {
    RunPrincipal& rp = *principalCache_.runPrincipalPtr();
    propagateProducts(InRun, principal, rp);
    typedef OccurrenceTraits<RunPrincipal, BranchActionGlobalEnd> Traits;
    schedule_->processOneOccurrence<Traits>(rp, esInfo_->es_, cleaningUpAfterException_);
    if(subProcess_.get()) subProcess_->doEndRun(rp, esInfo_->ts_, cleaningUpAfterException_);
  }

  void
  SubProcess::writeRun(ProcessHistoryID const& parentPhID, int runNumber) {
    ServiceRegistry::Operate operate(serviceToken_);
    std::map<ProcessHistoryID, ProcessHistoryID>::const_iterator it = parentToChildPhID_.find(parentPhID);
    assert(it != parentToChildPhID_.end());
    schedule_->writeRun(principalCache_.runPrincipal(it->second, runNumber));
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
    esInfo_.reset(new ESInfo(ts, *esp_));
    CurrentProcessingContext cpc;
    doBeginLuminosityBlock(principal, esInfo_->es_, &cpc);
    esInfo_.reset();
  }

  void
  SubProcess::beginLuminosityBlock(LuminosityBlockPrincipal const& principal) {
    boost::shared_ptr<LuminosityBlockAuxiliary> aux(new LuminosityBlockAuxiliary(principal.aux()));
    aux->setProcessHistoryID(principal.processHistoryID());
    boost::shared_ptr<LuminosityBlockPrincipal> lbpp(new LuminosityBlockPrincipal(aux, preg_, *processConfiguration_, historyAppender_.get(),principal.index()));
    lbpp->fillLuminosityBlockPrincipal(principal.reader());
    lbpp->setRunPrincipal(principalCache_.runPrincipalPtr());
    principalCache_.insert(lbpp);
    LuminosityBlockPrincipal& lbp = *principalCache_.lumiPrincipalPtr();
    propagateProducts(InLumi, principal, lbp);
    typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalBegin> Traits;
    schedule_->processOneOccurrence<Traits>(lbp, esInfo_->es_);
    if(subProcess_.get()) subProcess_->doBeginLuminosityBlock(lbp, esInfo_->ts_);
  }

  void
  SubProcess::doEndLuminosityBlock(LuminosityBlockPrincipal const& principal, IOVSyncValue const& ts, bool cleaningUpAfterException) {
    cleaningUpAfterException_ = cleaningUpAfterException;
    ServiceRegistry::Operate operate(serviceToken_);
    esInfo_.reset(new ESInfo(ts, *esp_));
    CurrentProcessingContext cpc;
    doEndLuminosityBlock(principal, esInfo_->es_, &cpc);
    esInfo_.reset();
  }

  void
  SubProcess::endLuminosityBlock(LuminosityBlockPrincipal const& principal) {
    LuminosityBlockPrincipal& lbp = *principalCache_.lumiPrincipalPtr();
    propagateProducts(InLumi, principal, lbp);
    typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionGlobalEnd> Traits;
    schedule_->processOneOccurrence<Traits>(lbp, esInfo_->es_, cleaningUpAfterException_);
    if(subProcess_.get()) subProcess_->doEndLuminosityBlock(lbp, esInfo_->ts_, cleaningUpAfterException_);
  }

  void
  SubProcess::writeLumi(ProcessHistoryID const& parentPhID, int runNumber, int lumiNumber) {
    ServiceRegistry::Operate operate(serviceToken_);
    std::map<ProcessHistoryID, ProcessHistoryID>::const_iterator it = parentToChildPhID_.find(parentPhID);
    assert(it != parentToChildPhID_.end());
    schedule_->writeLumi(principalCache_.lumiPrincipal(it->second, runNumber, lumiNumber));
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
  SubProcess::doBeginStream(StreamID iID) {
    ServiceRegistry::Operate operate(serviceToken_);
    assert(iID == schedule_->streamID());
    schedule_->beginStream();
    if(subProcess_.get()) subProcess_->doBeginStream(iID);
  }

  void
  SubProcess::doEndStream(StreamID iID) {
    ServiceRegistry::Operate operate(serviceToken_);
    assert(iID == schedule_->streamID());
    schedule_->endStream();
    if(subProcess_.get()) subProcess_->doEndStream(iID);
  }

  //Dummies until SubProcess inherits from new interface
  void
  SubProcess::doStreamBeginRun(StreamID, RunPrincipal const& principal, IOVSyncValue const& ts) {}
  
  void
  SubProcess::doStreamEndRun(StreamID, RunPrincipal const& principal, IOVSyncValue const& ts, bool cleaningUpAfterException) {}
  
  void
  SubProcess::doStreamBeginLuminosityBlock(StreamID, LuminosityBlockPrincipal const& principal, IOVSyncValue const& ts) {}
  
  void
  SubProcess::doStreamEndLuminosityBlock(StreamID, LuminosityBlockPrincipal const& principal, IOVSyncValue const& ts, bool cleaningUpAfterException) {}


  void
  SubProcess::propagateProducts(BranchType type, Principal const& parentPrincipal, Principal& principal) const {
    Selections const& keptVector = keptProducts()[type];
    for(Selections::const_iterator it = keptVector.begin(), itEnd = keptVector.end(); it != itEnd; ++it) {
      ProductHolderBase const* parentProductHolder = parentPrincipal.getProductHolder((*it)->branchID(), false, false);
      if(parentProductHolder != 0) {
        ProductData const& parentData = parentProductHolder->productData();
        ProductHolderBase const* productHolder = principal.getProductHolder((*it)->branchID(), false, false);
        if(productHolder != 0) {
          ProductData& thisData = const_cast<ProductData&>(productHolder->productData());
          //Propagate the per event(run)(lumi) data for this product to the subprocess.
          //First, the product itself.
          thisData.wrapper_ = parentData.wrapper_;
          // Then the product ID and the ProcessHistory 
          thisData.prov_.setProductID(parentData.prov_.productID());
          thisData.prov_.setProcessHistoryID(parentData.prov_.processHistoryID());
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

