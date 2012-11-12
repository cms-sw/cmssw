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

using namespace ecaldqm;

EcalDQMonitorTask::EcalDQMonitorTask(const edm::ParameterSet &_ps) :
  EcalDQMonitor(_ps),
  ievt_(0),
  workers_(0),
  schedule_(),
  enabled_(),
  taskTimes_(),
  evaluateTime_(_ps.getUntrackedParameter<bool>("evaluateTime")),
  allowMissingCollections_(_ps.getUntrackedParameter<bool>("allowMissingCollections"))
{
  using namespace std;

  vector<string> taskNames(_ps.getUntrackedParameter<vector<string> >("tasks"));
  const edm::ParameterSet& taskParams(_ps.getUntrackedParameterSet("taskParameters"));
  const edm::ParameterSet& mePaths(_ps.getUntrackedParameterSet("mePaths"));

  WorkerFactory factory(0);
  std::multimap<Collections, Collections> dependencies;

  for(vector<string>::iterator tItr(taskNames.begin()); tItr != taskNames.end(); ++tItr){
    if (!(factory = SetWorker::findFactory(*tItr))) continue;

    if(verbosity_ > 0) cout << moduleName_ << ": Setting up " << *tItr << endl;

    DQWorker* worker(factory(taskParams, mePaths.getUntrackedParameterSet(*tItr)));
    if(worker->getName() != *tItr){
      delete worker;

      if(verbosity_ > 0) cout << moduleName_ << ": " << *tItr << " could not be configured" << endl; 
      continue;
    }
    DQWorkerTask* task(static_cast<DQWorkerTask*>(worker));
    task->setVerbosity(verbosity_);

    workers_.push_back(task);

    const std::vector<std::pair<Collections, Collections> >& dep(task->getDependencies());
    for(std::vector<std::pair<Collections, Collections> >::const_iterator depItr(dep.begin()); depItr != dep.end(); ++depItr)
      dependencies.insert(*depItr);
  }

  const edm::ParameterSet& collectionTags(_ps.getUntrackedParameterSet("collectionTags"));

  std::vector<Collections> usedCollections;

  for(unsigned iCol(0); iCol < nCollections; iCol++){

    collectionTags_[iCol] = edm::InputTag();
    taskLists_[iCol] = vector<DQWorkerTask*>();

    bool use(iCol == kEcalRawData);

    for(vector<DQWorkerTask*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
      if((*wItr)->runsOn(iCol)){
	taskLists_[iCol].push_back(*wItr);
	use = true;
      }
    }
    if(use){
      collectionTags_[iCol] = collectionTags.getUntrackedParameter<edm::InputTag>(collectionName[iCol]);
      usedCollections.push_back(Collections(iCol));
    }

  }

  formSchedule_(usedCollections, dependencies);

  if(verbosity_ > 0){
    cout << moduleName_ << ": Using collections" << endl;
    for(unsigned iCol(0); iCol < schedule_.size(); iCol++)
      cout << collectionName[schedule_[iCol].second] << endl;
    cout << endl;
  }
}

EcalDQMonitorTask::~EcalDQMonitorTask()
{
  for(std::vector<DQWorkerTask *>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr)
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
EcalDQMonitorTask::beginRun(const edm::Run &_run, const edm::EventSetup &_es)
{
  // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
  edm::ESHandle<EcalElectronicsMapping> elecMapHandle;
  _es.get<EcalMappingRcd>().get(elecMapHandle);
  setElectronicsMap(elecMapHandle.product());

  // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
  edm::ESHandle<EcalTrigTowerConstituentsMap> ttMapHandle;
  _es.get<IdealGeometryRecord>().get(ttMapHandle);
  setTrigTowerMap(ttMapHandle.product());

  for(std::vector<DQWorkerTask*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    DQWorkerTask* task(*wItr);
    task->reset();
    if(task->runsOn(kRun)) task->beginRun(_run, _es);
  }

  if(verbosity_ > 0)
    std::cout << moduleName_ << ": Starting run " << _run.run() << std::endl;

  ievt_ = 0;
  taskTimes_.clear();
}

void
EcalDQMonitorTask::endRun(const edm::Run &_run, const edm::EventSetup &_es)
{
  using namespace std;

  for(vector<DQWorkerTask *>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    DQWorkerTask* task(*wItr);
    if(task->runsOn(kRun)) task->endRun(_run, _es);
  }

  if(evaluateTime_){
    stringstream ss;

    ss << "************** " << moduleName_ << " **************" << endl;
    ss << "      Mean time consumption of the modules" << endl;
    ss << "____________________________________" << endl;
    for(std::vector<DQWorkerTask *>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
      DQWorkerTask* task(*wItr);
      ss << setw(20) << setfill(' ') << task->getName() << "|   " << (taskTimes_[task] / ievt_) << endl;
    }
    edm::LogInfo("EcalDQM") << ss.str();
  }
}

void
EcalDQMonitorTask::beginLuminosityBlock(const edm::LuminosityBlock &_lumi, const edm::EventSetup &_es)
{
  for(std::vector<DQWorkerTask *>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    DQWorkerTask* task(*wItr);
    if(task->isInitialized() && task->runsOn(kLumiSection)) task->beginLuminosityBlock(_lumi, _es);
  }
}

void
EcalDQMonitorTask::endLuminosityBlock(const edm::LuminosityBlock &_lumi, const edm::EventSetup &_es)
{
  for(std::vector<DQWorkerTask *>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    DQWorkerTask* task(*wItr);
    if(task->isInitialized() && task->runsOn(kLumiSection)) task->endLuminosityBlock(_lumi, _es);
  }
}

void
EcalDQMonitorTask::analyze(const edm::Event &_evt, const edm::EventSetup &_es)
{
  using namespace std;
  using namespace ecaldqm;

  ievt_++;

  edm::Handle<EcalRawDataCollection> dcchsHndl;
  if(!_evt.getByLabel(collectionTags_[kEcalRawData], dcchsHndl))
    throw cms::Exception("ObjectNotFound") << "EcalRawDataCollection with InputTag " << collectionTags_[kEcalRawData] << " does not exist";
 
  // determine event type (called run type in DCCHeader for some reason) for each FED
  std::vector<short> runType(54, -1);
  for(EcalRawDataCollection::const_iterator dcchItr = dcchsHndl->begin(); dcchItr != dcchsHndl->end(); ++dcchItr){
    runType[dcchItr->id() - 1] = dcchItr->getRunType();
  }

  bool atLeastOne(false);

  DQWorkerTask *task(0);

  // set up task modules
  for(vector<DQWorkerTask*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    task = *wItr;

    if(task->filterRunType(runType)){
      enabled_[task] = true;

      if(!task->isInitialized()){
	if(verbosity_ > 1) cout << moduleName_ << ": Booking MEs for " << task->getName() << endl;
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
  for(vector<DQWorkerTask *>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    task = *wItr;
    if(enabled_[task]) task->endEvent(_evt, _es);
  }

}
