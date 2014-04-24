#include "../interface/EcalDQMonitorTask.h"

#include "../interface/DQWorkerTask.h"

#include "DQM/EcalCommon/interface/MESet.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include <iomanip>
#include <ctime>
#include <bitset>
#include <sstream>

EcalDQMonitorTask::EcalDQMonitorTask(edm::ParameterSet const& _ps) :
  DQMEDAnalyzer(),
  ecaldqm::EcalDQMonitor(_ps),
  schedule_(),
  allowMissingCollections_(_ps.getUntrackedParameter<bool>("allowMissingCollections")),
  processedEvents_(0),
  evaluateTime_(_ps.getUntrackedParameter<bool>("evaluateTime")),
  taskTimes_(),
  lastResetTime_(0),
  resetInterval_(_ps.getUntrackedParameter<double>("resetInterval"))
{
  ecaldqm::DependencySet dependencies;
  std::bitset<ecaldqm::nCollections> hasTaskToRun;
  edm::ConsumesCollector collector(consumesCollector());

  executeOnWorkers_([&dependencies, &hasTaskToRun, &collector, this](ecaldqm::DQWorker* worker){
                      ecaldqm::DQWorkerTask* task(dynamic_cast<ecaldqm::DQWorkerTask*>(worker));
                      if(!task)
                        throw cms::Exception("InvalidConfiguration") << "Non-task DQWorker " << worker->getName() << " passed";

                      task->addDependencies(dependencies);
                      for(unsigned iCol(0); iCol < ecaldqm::nCollections; ++iCol){
                        if(task->analyze(0, ecaldqm::Collections(iCol))) // "dry run" mode
                          hasTaskToRun.set(iCol);
                      }

                      task->setTokens(collector);

                      taskTimes_[task] = 0.;
                    }, "initialization");

  edm::ParameterSet const& collectionTags(_ps.getUntrackedParameterSet("collectionTags"));

  for(unsigned iCol(0); iCol < ecaldqm::nCollections; iCol++){
    if(hasTaskToRun[iCol])
      dependencies.push_back(ecaldqm::Dependency(ecaldqm::Collections(iCol)));
  }
  if(collectionTags.existsAs<edm::InputTag>("EcalRawData"))
    dependencies.push_back(ecaldqm::Dependency(ecaldqm::kEcalRawData));

  formSchedule(dependencies.formSequence(), collectionTags);

  if(verbosity_ > 0){
    std::stringstream ss;
    ss << moduleName_ << ": Using collections" << std::endl;
    for(unsigned iCol(0); iCol < schedule_.size(); iCol++)
      ss << ecaldqm::collectionName[schedule_[iCol].second] << std::endl;
    edm::LogInfo("EcalDQM") << ss.str();
  }

  edm::ParameterSet const& commonParams(_ps.getUntrackedParameterSet("commonParameters"));
  if(commonParams.getUntrackedParameter<bool>("onlineMode"))
    lastResetTime_ = time(0);
}

/*static*/
void
EcalDQMonitorTask::fillDescriptions(edm::ConfigurationDescriptions& _descs)
{
  edm::ParameterSetDescription desc;
  ecaldqm::EcalDQMonitor::fillDescriptions(desc);

  edm::ParameterSetDescription taskParameters;
  ecaldqm::DQWorkerTask::fillDescriptions(taskParameters);
  edm::ParameterSetDescription allWorkers;
  allWorkers.addNode(edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, false, taskParameters));
  desc.addUntracked("workerParameters", allWorkers);

  edm::ParameterSetDescription collectionTags;
  collectionTags.addWildcardUntracked<edm::InputTag>("*");
  desc.addUntracked("collectionTags", collectionTags);

  desc.addUntracked<bool>("allowMissingCollections", true);
  desc.addUntracked<bool>("evaluateTime", false);
  desc.addUntracked<double>("resetInterval", 0.);

  _descs.addDefault(desc);
}

void
EcalDQMonitorTask::bookHistograms(DQMStore::IBooker& _ibooker, edm::Run const&, edm::EventSetup const& _es)
{
  ecaldqmGetSetupObjects(_es);
  ecaldqmBookHistograms(_ibooker);
}

void
EcalDQMonitorTask::dqmBeginRun(edm::Run const& _run, edm::EventSetup const& _es)
{
  ecaldqmBeginRun(_run, _es);

  processedEvents_ = 0;

  if(lastResetTime_ != 0) lastResetTime_ = time(0);
}

void
EcalDQMonitorTask::endRun(edm::Run const& _run, edm::EventSetup const& _es)
{
  if(lastResetTime_ != 0)
    executeOnWorkers_([](ecaldqm::DQWorker* worker){
                        static_cast<ecaldqm::DQWorkerTask*>(worker)->recoverStats();
                      }, "recoverStats");

  ecaldqmEndRun(_run, _es);

  ecaldqmReleaseHistograms();

  if(evaluateTime_){
    std::stringstream ss;

    ss << "************** " << moduleName_ << " **************" << std::endl;
    ss << "      Mean time consumption of the modules" << std::endl;
    ss << "____________________________________" << std::endl;
    executeOnWorkers_([&ss, this](ecaldqm::DQWorker* worker){
                        ss << std::setw(20) << std::setfill(' ') << worker->getName() << "|   " << (this->taskTimes_[worker] / this->processedEvents_) << std::endl;
                      }, "print time");
    edm::LogInfo("EcalDQM") << ss.str();
  }
}

void
EcalDQMonitorTask::beginLuminosityBlock(edm::LuminosityBlock const& _lumi, edm::EventSetup const& _es)
{
  ecaldqmBeginLuminosityBlock(_lumi, _es);
}

void
EcalDQMonitorTask::endLuminosityBlock(edm::LuminosityBlock const& _lumi, edm::EventSetup const& _es)
{
  ecaldqmEndLuminosityBlock(_lumi, _es);
  
  if(lastResetTime_ != 0 && (time(0) - lastResetTime_) / 3600. > resetInterval_){
    if(verbosity_ > 0) edm::LogInfo("EcalDQM") << moduleName_ << ": Soft-resetting the histograms";
    executeOnWorkers_([](ecaldqm::DQWorker* worker){
                        static_cast<ecaldqm::DQWorkerTask*>(worker)->softReset();
                      }, "softReset");

    lastResetTime_ = time(0);
  }
}

void
EcalDQMonitorTask::analyze(edm::Event const& _evt, edm::EventSetup const& _es)
{
  if(verbosity_ > 2) edm::LogInfo("EcalDQM") << moduleName_ << "::analyze: Run " << _evt.id().run() << " Lumisection " << _evt.id().luminosityBlock() << " Event " << _evt.id().event() << ": processed " << processedEvents_;

  if(schedule_.size() == 0) return;

  std::set<ecaldqm::DQWorker*> enabledTasks;

  edm::Handle<EcalRawDataCollection> dcchsHndl;
  if(_evt.getByToken(collectionTokens_[ecaldqm::kEcalRawData], dcchsHndl)){
    // determine event type (called run type in DCCHeader for some reason) for each FED
    std::stringstream ss;
    if(verbosity_ > 2) ss << moduleName_ << ": Event type ";

    short runType[ecaldqm::nDCC];
    std::fill_n(runType, ecaldqm::nDCC, -1);
    for(EcalRawDataCollection::const_iterator dcchItr = dcchsHndl->begin(); dcchItr != dcchsHndl->end(); ++dcchItr){
      if(verbosity_ > 2) ss << dcchItr->getRunType() << " ";
      runType[dcchItr->id() - 1] = dcchItr->getRunType();
    }
    if(verbosity_ > 2) edm::LogInfo("EcalDQM") << ss.str();

    bool processEvent(false);

    executeOnWorkers_([&enabledTasks, &runType, &processEvent, this](ecaldqm::DQWorker* worker){
                        if(static_cast<ecaldqm::DQWorkerTask*>(worker)->filterRunType(runType)){
                          if(this->verbosity_ > 2) edm::LogInfo("EcalDQM") << worker->getName() << " will run on this event";
                          enabledTasks.insert(worker);
                          processEvent = true;
                        }
                      }, "filterRunType");

    if(!processEvent) return;
  }
  else{
    edm::LogWarning("EcalDQM") << "EcalRawDataCollection does not exist. No event-type filtering will be applied";
    executeOnWorkers_([&enabledTasks](ecaldqm::DQWorker* worker){
                        enabledTasks.insert(worker);
                      }, "");
  }

  ++processedEvents_;

  // start event processing
  executeOnWorkers_([&_evt, &_es, &enabledTasks](ecaldqm::DQWorker* worker){
                      if(enabledTasks.find(worker) != enabledTasks.end()){
                        if(worker->onlineMode()) worker->setTime(time(0));
                        worker->setEventNumber(_evt.id().event());
                        static_cast<ecaldqm::DQWorkerTask*>(worker)->beginEvent(_evt, _es);
                      }
                    }, "beginEvent");

  // run on collections
  for(unsigned iSch(0); iSch < schedule_.size(); iSch++){
    Processor processor(schedule_[iSch].first);
    (this->*processor)(_evt, schedule_[iSch].second, enabledTasks);
  }

  // close event processing
  executeOnWorkers_([&_evt, &_es, &enabledTasks](ecaldqm::DQWorker* worker){
                      if(enabledTasks.find(worker) != enabledTasks.end())
                        static_cast<ecaldqm::DQWorkerTask*>(worker)->endEvent(_evt, _es);
                    }, "endEvent");

  if(verbosity_ > 2) edm::LogInfo("EcalDQM") << moduleName_ << "::analyze: Closing Event " << _evt.id().event();
}

DEFINE_FWK_MODULE(EcalDQMonitorTask);
