#include "FWCore/Framework/interface/SubProcess.h"

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "FWCore/Framework/interface/Group.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/src/EventSetupsController.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <cassert>
#include <set>
#include <string>
#include <vector>

namespace edm {

  SubProcess::SubProcess(ParameterSet& parameterSet,
                         ParameterSet const& topLevelParameterSet,
                         boost::shared_ptr<ProductRegistry const> parentProductRegistry,
                         eventsetup::EventSetupsController& esController,
                         ActivityRegistry& parentActReg,
                         ServiceToken const& token,
                         serviceregistry::ServiceLegacy iLegacy) :
      OutputModule(parameterSet),
      serviceToken_(),
      parentPreg_(parentProductRegistry),
      preg_(),
      act_table_(),
      processConfiguration_(),
      principalCache_(),
      esp_(),
      schedule_(),
      parentToChildPhID_(),
      esInfo_(0),
      subProcess_() {

    std::string const maxEvents("maxEvents");
    std::string const maxLumis("maxLuminosityBlocks");

    std::auto_ptr<ParameterSet> processParameterSet = parameterSet.popParameterSet(std::string("process")); 

    // if this process has a maxEvents or maxLuminosityBlocks parameter set, remove them.
    if(processParameterSet->exists(maxEvents)) {
      processParameterSet->popParameterSet(maxEvents);
    }
    if(processParameterSet->exists(maxLumis)) {
      processParameterSet->popParameterSet(maxLumis);
    }

    // if the top level process has a maxEvents or maxLuminosityBlocks parameter set, add them to this process.
    if(topLevelParameterSet.exists(maxEvents)) {
      processParameterSet->addUntrackedParameter<ParameterSet>(maxEvents, topLevelParameterSet.getUntrackedParameterSet(maxEvents));
    }
    if(topLevelParameterSet.exists(maxLumis)) {
      processParameterSet->addUntrackedParameter<ParameterSet>(maxLumis, topLevelParameterSet.getUntrackedParameterSet(maxLumis));
    }

    // If this process has a subprocess, pop the subprocess parameter set out of the process parameter set

    boost::shared_ptr<ParameterSet> subProcessParameterSet(popSubProcessParameterSet(*processParameterSet).release());
  
    ScheduleItems items(*parentProductRegistry);

    //initialize the services
    ServiceToken iToken;

    // get any configured services.
    std::auto_ptr<std::vector<ParameterSet> > serviceSets = processParameterSet->popVParameterSet(std::string("services")); 

    ServiceToken newToken = items.initServices(*serviceSets, *processParameterSet, token, iLegacy, false);
    parentActReg.connectToSubProcess(*items.actReg_);
    serviceToken_ = items.addCPRandTNS(*processParameterSet, newToken);


    //make the services available
    ServiceRegistry::Operate operate(serviceToken_);

    // intialize miscellaneous items
    boost::shared_ptr<CommonParams> common(items.initMisc(*processParameterSet));

    // intialize the event setup provider
    esp_ = esController.makeProvider(*processParameterSet, *common);

    // intialize the Schedule
    schedule_ = items.initSchedule(*processParameterSet);

    // set the items
    act_table_ = items.act_table_;
    preg_ = items.preg_;
    processConfiguration_ = items.processConfiguration_;

    boost::shared_ptr<EventPrincipal> ep(new EventPrincipal(preg_, *processConfiguration_));
    principalCache_.insert(ep);

    if(subProcessParameterSet) {
      subProcess_.reset(new SubProcess(*subProcessParameterSet, topLevelParameterSet, preg_, esController, *items.actReg_, newToken, iLegacy));
    }
  }

  SubProcess::~SubProcess() {}

  void
  SubProcess::beginJob() {
    // Mark dropped branches as dropped in the product registry.
    {
      std::set<BranchID> keptBranches;
      Selections const& keptVectorR = keptProducts()[InRun];
      for(Selections::const_iterator it = keptVectorR.begin(), itEnd = keptVectorR.end(); it != itEnd; ++it) {
        keptBranches.insert((*it)->branchID());
      }
      Selections const& keptVectorL = keptProducts()[InLumi];
      for(Selections::const_iterator it = keptVectorL.begin(), itEnd = keptVectorL.end(); it != itEnd; ++it) {
        keptBranches.insert((*it)->branchID());
      }
      Selections const& keptVectorE = keptProducts()[InEvent];
      for(Selections::const_iterator it = keptVectorE.begin(), itEnd = keptVectorE.end(); it != itEnd; ++it) {
        keptBranches.insert((*it)->branchID());
      }
      for(ProductRegistry::ProductList::const_iterator it = preg_->productList().begin(), itEnd = preg_->productList().end(); it != itEnd; ++it) {
        if(keptBranches.find(it->second.branchID()) == keptBranches.end()) {
          it->second.setDropped();
        } 
      }
    }
    ServiceRegistry::Operate operate(serviceToken_);
    schedule_->beginJob();
    if(subProcess_.get()) subProcess_->doBeginJob();
  }

  void
  SubProcess::endJob() {
    ServiceRegistry::Operate operate(serviceToken_);
    schedule_->endJob();
    if(subProcess_.get()) subProcess_->doEndJob();
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
    std::auto_ptr<EventAuxiliary> aux(new EventAuxiliary(principal.aux()));
    aux->setProcessHistoryID(principal.processHistoryID());
    EventPrincipal& ep = principalCache_.eventPrincipal();
    ep.fillEventPrincipal(aux,
                          principalCache_.lumiPrincipalPtr(),
                          boost::shared_ptr<EventSelectionIDVector>(new EventSelectionIDVector), // UGH use subprocess specific parameter
                          boost::shared_ptr<BranchListIndexes>(new BranchListIndexes), // UGH figure this out
                          principal.branchMapperPtr(),
                          principal.store());
    propagateProducts(InEvent, principal, ep);
    typedef OccurrenceTraits<EventPrincipal, BranchActionBegin> Traits;
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
    boost::shared_ptr<RunPrincipal> rpp(new RunPrincipal(aux, preg_, *processConfiguration_));
    rpp->fillRunPrincipal(principal.branchMapperPtr(), principal.store());
    principalCache_.insert(rpp);
    parentToChildPhID_.insert(std::make_pair(principal.aux().processHistoryID(), principal.processHistoryID()));
    parentToChildPhID_.insert(std::make_pair(principal.processHistoryID(), rpp->processHistoryID()));
    RunPrincipal& rp = *principalCache_.runPrincipalPtr();
    propagateProducts(InRun, principal, rp);
    typedef OccurrenceTraits<RunPrincipal, BranchActionBegin> Traits;
    schedule_->processOneOccurrence<Traits>(rp, esInfo_->es_);
    if(subProcess_.get()) subProcess_->doBeginRun(rp, esInfo_->ts_);
  }

  void
  SubProcess::doEndRun(RunPrincipal const& principal, IOVSyncValue const& ts) {
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
    typedef OccurrenceTraits<RunPrincipal, BranchActionEnd> Traits;
    schedule_->processOneOccurrence<Traits>(rp, esInfo_->es_);
    if(subProcess_.get()) subProcess_->doEndRun(rp, esInfo_->ts_);
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
    boost::shared_ptr<LuminosityBlockPrincipal> lbpp(new LuminosityBlockPrincipal(aux, preg_, *processConfiguration_, principalCache_.runPrincipalPtr()));
    lbpp->fillLuminosityBlockPrincipal(principal.branchMapperPtr(), principal.store());
    principalCache_.insert(lbpp);
    parentToChildPhID_.insert(std::make_pair(principal.aux().processHistoryID(), principal.processHistoryID()));
    parentToChildPhID_.insert(std::make_pair(principal.processHistoryID(), lbpp->processHistoryID()));
    LuminosityBlockPrincipal& lbp = *principalCache_.lumiPrincipalPtr();
    propagateProducts(InLumi, principal, lbp);
    typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionBegin> Traits;
    schedule_->processOneOccurrence<Traits>(lbp, esInfo_->es_);
    if(subProcess_.get()) subProcess_->doBeginLuminosityBlock(lbp, esInfo_->ts_);
  }

  void
  SubProcess::doEndLuminosityBlock(LuminosityBlockPrincipal const& principal, IOVSyncValue const& ts) {
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
    typedef OccurrenceTraits<LuminosityBlockPrincipal, BranchActionEnd> Traits;
    schedule_->processOneOccurrence<Traits>(lbp, esInfo_->es_);
    if(subProcess_.get()) subProcess_->doEndLuminosityBlock(lbp, esInfo_->ts_);
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
  SubProcess::propagateProducts(BranchType type, Principal const& parentPrincipal, Principal& principal) const {
    Selections const& keptVector = keptProducts()[type];
    for(Selections::const_iterator it = keptVector.begin(), itEnd = keptVector.end(); it != itEnd; ++it) {
      boost::shared_ptr<Group const> parentGroup = parentPrincipal.getGroup((*it)->branchID(), false, false);
      if(parentGroup) {
        // Make copy of parent group data
        GroupData parentData = parentGroup->groupData();
        boost::shared_ptr<Group const> group = principal.getGroup((*it)->branchID(), false, false);
        if(group) {
          // Swap copy with this group data
          GroupData& thisData = const_cast<GroupData&>(group->groupData());
          thisData.swap(parentData);
          // Sets unavailable flag, if known that product is not available
          group->productUnavailable();
        }
      }
    }
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
    return std::auto_ptr<ParameterSet>(0);
  }

}

