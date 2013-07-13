#include "FWCore/Framework/interface/ScheduleItems.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/Selections.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/interface/CommonParams.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include <set>

namespace edm {
  ScheduleItems::ScheduleItems() :
      actReg_(new ActivityRegistry),
      preg_(new SignallingProductRegistry),
      branchIDListHelper_(new BranchIDListHelper()),
      act_table_(),
      processConfiguration_() {
  }

  ScheduleItems::ScheduleItems(ProductRegistry const& preg, BranchIDListHelper const& branchIDListHelper, OutputModule const& om) :
      actReg_(new ActivityRegistry),
      preg_(new SignallingProductRegistry(preg)),
      branchIDListHelper_(new BranchIDListHelper(branchIDListHelper)),
      act_table_(),
      processConfiguration_() {

    for(auto& item : preg_->productListUpdator()) {
      BranchDescription& prod = item.second;
      prod.setOnDemand(false);
      prod.setProduced(false);
    }

    // Mark dropped branches as dropped in the product registry.
    std::set<BranchID> keptBranches;
    Selections const& keptVectorR = om.keptProducts()[InRun];
    for(auto const& item : keptVectorR) {
      keptBranches.insert(item->branchID());
    }
    Selections const& keptVectorL = om.keptProducts()[InLumi];
    for(auto const& item : keptVectorL) {
      keptBranches.insert(item->branchID());
    }
    Selections const& keptVectorE = om.keptProducts()[InEvent];
    for(auto const& item : keptVectorE) {
      keptBranches.insert(item->branchID());
    }
    for(auto& item : preg_->productListUpdator()) {
      BranchDescription& prod = item.second;
      if(keptBranches.find(prod.branchID()) == keptBranches.end()) {
        prod.setDropped(true);
      }
    }
  }

  ServiceToken
  ScheduleItems::initServices(std::vector<ParameterSet>& pServiceSets,
                             ParameterSet& parameterSet,
                             ServiceToken const& iToken,
                             serviceregistry::ServiceLegacy iLegacy,
                             bool associate) {

    //create the services
    ServiceToken token(ServiceRegistry::createSet(pServiceSets, iToken, iLegacy, associate));

    //see if any of the Services have to have their PSets stored
    for(auto const& item : pServiceSets) {
      if(item.exists("@save_config")) {
        parameterSet.addParameter(item.getParameter<std::string>("@service_type"), item);
      }
    }
    // Copy slots that hold all the registered callback functions like
    // PostBeginJob into an ActivityRegistry
    token.copySlotsTo(*actReg_);
    return token;
  }

  ServiceToken
  ScheduleItems::addCPRandTNS(ParameterSet const& parameterSet, ServiceToken const& token) {

    //add the ProductRegistry as a service ONLY for the construction phase
    typedef serviceregistry::ServiceWrapper<ConstProductRegistry> w_CPR;
    boost::shared_ptr<w_CPR>
      reg(new w_CPR(std::auto_ptr<ConstProductRegistry>(new ConstProductRegistry(*preg_))));
    ServiceToken tempToken(ServiceRegistry::createContaining(reg,
                                                             token,
                                                             serviceregistry::kOverlapIsError));

    // the next thing is ugly: pull out the trigger path pset and
    // create a service and extra token for it

    typedef service::TriggerNamesService TNS;
    typedef serviceregistry::ServiceWrapper<TNS> w_TNS;

    boost::shared_ptr<w_TNS> tnsptr
      (new w_TNS(std::auto_ptr<TNS>(new TNS(parameterSet))));

    return ServiceRegistry::createContaining(tnsptr,
                                             tempToken,
                                             serviceregistry::kOverlapIsError);
  }

  boost::shared_ptr<CommonParams>
  ScheduleItems::initMisc(ParameterSet& parameterSet) {
    act_table_.reset(new ActionTable(parameterSet));
    std::string processName = parameterSet.getParameter<std::string>("@process_name");
    processConfiguration_.reset(new ProcessConfiguration(processName, getReleaseVersion(), getPassID()));
    boost::shared_ptr<CommonParams>
        common(new CommonParams(parameterSet.getUntrackedParameterSet(
                                   "maxEvents", ParameterSet()).getUntrackedParameter<int>("input", -1),
                                parameterSet.getUntrackedParameterSet(
                                   "maxLuminosityBlocks", ParameterSet()).getUntrackedParameter<int>("input", -1)));
    return common;
  }

  std::auto_ptr<Schedule>
  ScheduleItems::initSchedule(ParameterSet& parameterSet,
                              ParameterSet const* subProcessPSet) {
    std::auto_ptr<Schedule> schedule(
        new Schedule(parameterSet,
                     ServiceRegistry::instance().get<service::TriggerNamesService>(),
                     *preg_,
                     *branchIDListHelper_,
                     *act_table_,
                     actReg_,
                     processConfiguration_,
                     subProcessPSet));
    return schedule;
  }

  void
  ScheduleItems::clear() {
    actReg_.reset();
    preg_.reset();
    branchIDListHelper_.reset();
    processConfiguration_.reset();
  }
}

