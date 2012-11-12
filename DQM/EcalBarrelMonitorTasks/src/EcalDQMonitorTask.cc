#include "../interface/EcalDQMonitorTask.h"

#include "../interface/DQWorkerTask.h"

#include <algorithm>
#include <iomanip>
#include <ctime>

#include "DQM/EcalCommon/interface/MESet.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
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
  processedEvents_(0),
  schedule_(),
  enabled_(),
  taskTimes_(),
  evaluateTime_(_ps.getUntrackedParameter<bool>("evaluateTime", false)),
  allowMissingCollections_(_ps.getUntrackedParameter<bool>("allowMissingCollections", false)),
  lastResetTime_(0),
  resetInterval_(0.)
{
  using namespace std;

  for(vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr)
    if(!dynamic_cast<DQWorkerTask*>(*wItr))
      throw cms::Exception("InvalidConfiguration") << "Non-client DQWorker " << (*wItr)->getName() << " passed";

  const edm::ParameterSet& collectionTags(_ps.getUntrackedParameterSet("collectionTags"));

  ecaldqm::DependencySet dependencies;
  bool atLeastOne(false);
  for(unsigned iCol(0); iCol < nCollections; iCol++){

    collectionTags_[iCol] = edm::InputTag();
    taskLists_[iCol] = vector<DQWorkerTask*>();

    bool use(false);

    for(vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
      DQWorkerTask* task(static_cast<DQWorkerTask*>(*wItr));

      task->setDependencies(dependencies);

      if(task->runsOn(iCol)){
	taskLists_[iCol].push_back(task);
	use = true;
      }
    }
    if(use){
      collectionTags_[iCol] = collectionTags.getUntrackedParameter<edm::InputTag>(collectionName[iCol]);
      dependencies.push_back(ecaldqm::Dependency(Collections(iCol)));
      atLeastOne = true;
    }
  }

  if(atLeastOne && collectionTags_[kEcalRawData] == edm::InputTag()){
    collectionTags_[kEcalRawData] = collectionTags.getUntrackedParameter<edm::InputTag>(collectionName[kEcalRawData]);
    dependencies.push_back(ecaldqm::Dependency(kEcalRawData));
  }

  formSchedule_(dependencies.formSequence());

  if(verbosity_ > 0){
    cout << moduleName_ << ": Using collections" << endl;
    for(unsigned iCol(0); iCol < schedule_.size(); iCol++)
      cout << collectionName[schedule_[iCol].second] << endl;
    cout << endl;
  }

  if(online_) resetInterval_ = _ps.getUntrackedParameter<double>("resetInterval");
}

EcalDQMonitorTask::~EcalDQMonitorTask()
{
  for(std::vector<DQWorker *>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr)
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
  DQWorker::iRun = _run.run();
  DQWorker::now = time(0);

  // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
  edm::ESHandle<EcalElectronicsMapping> elecMapHandle;
  _es.get<EcalMappingRcd>().get(elecMapHandle);
  setElectronicsMap(elecMapHandle.product());

  // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
  edm::ESHandle<EcalTrigTowerConstituentsMap> ttMapHandle;
  _es.get<IdealGeometryRecord>().get(ttMapHandle);
  setTrigTowerMap(ttMapHandle.product());

  for(std::vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    DQWorkerTask* task(static_cast<DQWorkerTask*>(*wItr));
    task->reset();
    if(task->runsOn(kRun)) task->beginRun(_run, _es);
  }

  if(verbosity_ > 0)
    std::cout << moduleName_ << ": Starting run " << _run.run() << std::endl;

  processedEvents_ = 0;
  taskTimes_.clear();

  if(online_) lastResetTime_ = time(0);
}

void
EcalDQMonitorTask::endRun(const edm::Run &_run, const edm::EventSetup &_es)
{
  using namespace std;

  for(vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    DQWorkerTask* task(static_cast<DQWorkerTask*>(*wItr));
    if(online_) task->recoverStats();
    if(task->runsOn(kRun)) task->endRun(_run, _es);
  }

  if(evaluateTime_){
    stringstream ss;

    ss << "************** " << moduleName_ << " **************" << endl;
    ss << "      Mean time consumption of the modules" << endl;
    ss << "____________________________________" << endl;
    for(std::vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
      DQWorker* worker(*wItr);
      ss << setw(20) << setfill(' ') << worker->getName() << "|   " << (taskTimes_[static_cast<DQWorkerTask*>(worker)] / processedEvents_) << endl;
    }
    edm::LogInfo("EcalDQM") << ss.str();
  }
}

void
EcalDQMonitorTask::beginLuminosityBlock(const edm::LuminosityBlock &_lumi, const edm::EventSetup &_es)
{
  DQWorker::iLumi = _lumi.luminosityBlock();
  DQWorker::now = time(0);

  for(std::vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    DQWorkerTask* task(static_cast<DQWorkerTask*>(*wItr));
    if(task->isInitialized() && task->runsOn(kLumiSection)) task->beginLuminosityBlock(_lumi, _es);
  }
}

void
EcalDQMonitorTask::endLuminosityBlock(const edm::LuminosityBlock &_lumi, const edm::EventSetup &_es)
{
  for(std::vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    DQWorkerTask* task(static_cast<DQWorkerTask*>(*wItr));
    if(task->isInitialized() && task->runsOn(kLumiSection)) task->endLuminosityBlock(_lumi, _es);
  }

  if(online_ && (time(0) - lastResetTime_) / 3600. > resetInterval_){
    if(verbosity_ > 0) std::cout << moduleName_ << ": Soft-resetting the histograms" << std::endl;
    for(std::vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
      DQWorkerTask* task(static_cast<DQWorkerTask*>(*wItr));
      task->softReset();
    }
    lastResetTime_ = time(0);
  }
}

void
EcalDQMonitorTask::analyze(const edm::Event &_evt, const edm::EventSetup &_es)
{
  using namespace std;
  using namespace ecaldqm;

  DQWorker::iEvt = _evt.id().event();
  DQWorker::now = time(0);
  processedEvents_++;

  if(verbosity_ > 2)
    std::cout << "Run " << DQWorker::iRun << " Lumisection " << DQWorker::iLumi << " Event " << DQWorker::iEvt << ": processed " << processedEvents_ << std::endl;

  if(schedule_.size() == 0) return;

  edm::Handle<EcalRawDataCollection> dcchsHndl;
  if(!_evt.getByLabel(collectionTags_[kEcalRawData], dcchsHndl))
    throw cms::Exception("ObjectNotFound") << "EcalRawDataCollection with InputTag " << collectionTags_[kEcalRawData] << " does not exist";
 
  // determine event type (called run type in DCCHeader for some reason) for each FED
  std::vector<short> runType(54, -1);
  for(EcalRawDataCollection::const_iterator dcchItr = dcchsHndl->begin(); dcchItr != dcchsHndl->end(); ++dcchItr)
    runType[dcchItr->id() - 1] = dcchItr->getRunType();

  bool atLeastOne(false);

  DQWorkerTask *task(0);

  // set up task modules
  for(vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    task = static_cast<DQWorkerTask*>(*wItr);

    if(task->filterRunType(runType)){
      enabled_[task] = true;

      if(!task->isInitialized()){
	task->initialize();
	if(verbosity_ > 1) cout << moduleName_ << ": Booking MEs for " << task->getName() << endl;
	task->bookMEs();
      }

      task->beginEvent(_evt, _es);
      atLeastOne = true;
    }
    else
      enabled_[task] = false;
  }

  if(!atLeastOne) return;

  // run on collections
  for(unsigned iSch(0); iSch < schedule_.size(); iSch++){

    Processor processor(schedule_[iSch].first);
    (this->*processor)(_evt, schedule_[iSch].second);
  }

  // close event
  for(vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    task = static_cast<DQWorkerTask*>(*wItr);
    if(enabled_[task]) task->endEvent(_evt, _es);
  }
}

DEFINE_FWK_MODULE(EcalDQMonitorTask);
