#include "DQM/EcalCommon/interface/EcalDQMonitorTask.h"

#include <algorithm>
#include <iomanip>

#include "DQM/EcalCommon/interface/DQWorkerTask.h"
#include "DQM/EcalCommon/interface/MESet.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

EcalDQMonitorTask::EcalDQMonitorTask(edm::ParameterSet const& _ps) :
  EcalDQMonitor(_ps),
  ievt_(0),
  workers_(0),
  schedule_(),
  enabled_(),
  taskTimes_(),
  evaluateTime_(_ps.getUntrackedParameter<bool>("evaluateTime")),
  allowMissingCollections_(_ps.getUntrackedParameter<bool>("allowMissingCollections"))
{
  std::vector<std::string> taskNames(_ps.getUntrackedParameter<std::vector<std::string> >("tasks"));
  edm::ParameterSet const& taskParams(_ps.getUntrackedParameterSet("taskParameters"));
  edm::ParameterSet const& mePaths(_ps.getUntrackedParameterSet("mePaths"));

  ecaldqm::WorkerFactory factory(0);
  std::multimap<ecaldqm::Collections, ecaldqm::Collections> dependencies;

  for(std::vector<std::string>::iterator tItr(taskNames.begin()); tItr != taskNames.end(); ++tItr){
    if(!(factory = ecaldqm::SetWorker::findFactory(*tItr))) continue;

    if(verbosity_ > 0) std::cout << moduleName_ << ": Setting up " << *tItr << std::endl;

    ecaldqm::DQWorker* worker(factory(taskParams, mePaths.getUntrackedParameterSet(*tItr)));
    if(worker->getName() != *tItr){
      delete worker;

      if(verbosity_ > 0) std::cout << moduleName_ << ": " << *tItr << " could not be configured" << std::endl; 
      continue;
    }
    ecaldqm::DQWorkerTask* task(static_cast<ecaldqm::DQWorkerTask*>(worker));
    task->setVerbosity(verbosity_);

    workers_.push_back(task);

    std::vector<std::pair<ecaldqm::Collections, ecaldqm::Collections> > const& dep(task->getDependencies());
    for(std::vector<std::pair<ecaldqm::Collections, ecaldqm::Collections> >::const_iterator depItr(dep.begin()); depItr != dep.end(); ++depItr)
      dependencies.insert(*depItr);
  }

  edm::ParameterSet const& collectionTags(_ps.getUntrackedParameterSet("collectionTags"));

  std::vector<ecaldqm::Collections> usedCollections;

  for(unsigned iCol(0); iCol < ecaldqm::nCollections; iCol++){

    collectionTokens_[iCol] = edm::EDGetToken();
    taskLists_[iCol] = std::vector<ecaldqm::DQWorkerTask*>();

    bool use(iCol == ecaldqm::kEcalRawData);

    for(std::vector<ecaldqm::DQWorkerTask*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
      if((*wItr)->runsOn(iCol)){
	taskLists_[iCol].push_back(*wItr);
	use = true;
      }
    }
    if(use){
      registerCollection(ecaldqm::Collections(iCol), collectionTags.getUntrackedParameter<edm::InputTag>(ecaldqm::collectionName[iCol]));
      usedCollections.push_back(ecaldqm::Collections(iCol));
    }

  }

  formSchedule_(usedCollections, dependencies);

  if(verbosity_ > 0){
    std::cout << moduleName_ << ": Using collections" << std::endl;
    for(unsigned iCol(0); iCol < schedule_.size(); iCol++)
      std::cout << ecaldqm::collectionName[schedule_[iCol].second] << std::endl;
    std::cout << std::endl;
  }
}

EcalDQMonitorTask::~EcalDQMonitorTask()
{
  for(std::vector<ecaldqm::DQWorkerTask *>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr)
    delete *wItr;
}

/* static */
void
EcalDQMonitorTask::fillDescriptions(edm::ConfigurationDescriptions &_descs)
{
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  _descs.addDefault(desc);
}

void
EcalDQMonitorTask::beginRun(edm::Run const& _run, edm::EventSetup const& _es)
{
  // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
  edm::ESHandle<EcalElectronicsMapping> elecMapHandle;
  _es.get<EcalMappingRcd>().get(elecMapHandle);
  ecaldqm::setElectronicsMap(elecMapHandle.product());

  // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
  edm::ESHandle<EcalTrigTowerConstituentsMap> ttMapHandle;
  _es.get<IdealGeometryRecord>().get(ttMapHandle);
  ecaldqm::setTrigTowerMap(ttMapHandle.product());

  for(std::vector<ecaldqm::DQWorkerTask*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    ecaldqm::DQWorkerTask* task(*wItr);
    task->reset();
    if(task->runsOn(ecaldqm::kRun)) task->beginRun(_run, _es);
  }

  if(verbosity_ > 0)
    std::cout << moduleName_ << ": Starting run " << _run.run() << std::endl;

  ievt_ = 0;
  taskTimes_.clear();
}

void
EcalDQMonitorTask::endRun(edm::Run const& _run, edm::EventSetup const& _es)
{
  for(std::vector<ecaldqm::DQWorkerTask*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    ecaldqm::DQWorkerTask* task(*wItr);
    if(task->runsOn(ecaldqm::kRun)) task->endRun(_run, _es);
  }

  if(evaluateTime_){
    std::stringstream ss;

    ss << "************** " << moduleName_ << " **************" << std::endl;
    ss << "      Mean time consumption of the modules" << std::endl;
    ss << "____________________________________" << std::endl;
    for(std::vector<ecaldqm::DQWorkerTask*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
      ecaldqm::DQWorkerTask* task(*wItr);
      ss << std::setw(20) << std::setfill(' ') << task->getName() << "|   " << (taskTimes_[task] / ievt_) << std::endl;
    }
    edm::LogInfo("EcalDQM") << ss.str();
  }
}

void
EcalDQMonitorTask::beginLuminosityBlock(edm::LuminosityBlock const& _lumi, edm::EventSetup const& _es)
{
  for(std::vector<ecaldqm::DQWorkerTask*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    ecaldqm::DQWorkerTask* task(*wItr);
    if(task->isInitialized() && task->runsOn(ecaldqm::kLumiSection)) task->beginLuminosityBlock(_lumi, _es);
  }
}

void
EcalDQMonitorTask::endLuminosityBlock(edm::LuminosityBlock const& _lumi, edm::EventSetup const& _es)
{
  for(std::vector<ecaldqm::DQWorkerTask*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    ecaldqm::DQWorkerTask* task(*wItr);
    if(task->isInitialized() && task->runsOn(ecaldqm::kLumiSection)) task->endLuminosityBlock(_lumi, _es);
  }
}

void
EcalDQMonitorTask::analyze(edm::Event const& _evt, edm::EventSetup const& _es)
{
  ievt_++;

  edm::Handle<EcalRawDataCollection> dcchsHndl;
  if(!_evt.getByToken(collectionTokens_[ecaldqm::kEcalRawData], dcchsHndl))
    throw cms::Exception("ObjectNotFound") << "EcalRawDataCollection does not exist";
 
  // determine event type (called run type in DCCHeader for some reason) for each FED
  std::vector<short> runType(54, -1);
  for(EcalRawDataCollection::const_iterator dcchItr = dcchsHndl->begin(); dcchItr != dcchsHndl->end(); ++dcchItr){
    runType[dcchItr->id() - 1] = dcchItr->getRunType();
  }

  bool atLeastOne(false);

  ecaldqm::DQWorkerTask* task(0);

  // set up task modules
  for(std::vector<ecaldqm::DQWorkerTask*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    task = *wItr;

    if(task->filterRunType(runType)){
      enabled_[task] = true;

      if(!task->isInitialized()){
	if(verbosity_ > 1) std::cout << moduleName_ << ": Booking MEs for " << task->getName() << std::endl;
	task->bookMEs();
	task->setInitialized(true);
      }

      task->beginEvent(_evt, _es);
      atLeastOne = true;
    }else{
      enabled_[task] = false;
    }
  }

  if(!atLeastOne) return;

  // run on collections
  for(unsigned iSch(0); iSch < schedule_.size(); iSch++){
    Processor processor(schedule_[iSch].first);
    (this->*processor)(_evt, schedule_[iSch].second);
  }

  // close event
  for(std::vector<ecaldqm::DQWorkerTask *>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    task = *wItr;
    if(enabled_[task]) task->endEvent(_evt, _es);
  }

}
