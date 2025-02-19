#include "FWCore/Framework/interface/ScheduleItems.h"

#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/interface/CommonParams.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

namespace edm {
  ScheduleItems::ScheduleItems() :
      actReg_(new ActivityRegistry),
      preg_(new SignallingProductRegistry),
      act_table_(),
      processConfiguration_() {
  }

  ScheduleItems::ScheduleItems(ProductRegistry const& preg) :
      actReg_(new ActivityRegistry),
      preg_(new SignallingProductRegistry(preg)),
      act_table_(),
      processConfiguration_() {
    for(ProductRegistry::ProductList::iterator it = preg_->productListUpdator().begin(), itEnd = preg_->productListUpdator().end(); it != itEnd; ++it) {
      it->second.onDemand() = false;
      it->second.produced() = false;
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
    for(std::vector<ParameterSet>::const_iterator it = pServiceSets.begin(), itEnd = pServiceSets.end();
        it != itEnd;
        ++it) {
      if(it->exists("@save_config")) {
        parameterSet.addParameter(it->getParameter<std::string>("@service_type"), *it);
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
    processConfiguration_.reset();
  }
}

