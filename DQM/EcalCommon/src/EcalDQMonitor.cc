#include "DQM/EcalCommon/interface/EcalDQMonitor.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

#include <sstream>
#include <ctime>

namespace ecaldqm
{
  EcalDQMonitor::EcalDQMonitor(edm::ParameterSet const& _ps) :
    workers_(),
    moduleName_(_ps.getUntrackedParameter<std::string>("moduleName")),
    verbosity_(_ps.getUntrackedParameter<int>("verbosity"))
  {
    std::vector<std::string> workerNames(_ps.getUntrackedParameter<std::vector<std::string> >("workers"));
    edm::ParameterSet const& workerParams(_ps.getUntrackedParameterSet("workerParameters"));
    edm::ParameterSet const& commonParams(_ps.getUntrackedParameterSet("commonParameters"));

    std::for_each(workerNames.begin(), workerNames.end(), [&](std::string const& name){
        if(verbosity_ > 0) edm::LogInfo("EcalDQM") << moduleName_ << ": Setting up " << name << std::endl;
        try{
          DQWorker* worker(WorkerFactoryStore::singleton()->getWorker(name, verbosity_, commonParams, workerParams.getUntrackedParameterSet(name)));
          if(worker->onlineMode()) worker->setTime(time(0));
          workers_.push_back(worker);
        }
        catch(std::exception&){
          edm::LogError("EcalDQM") << "Worker " << name << " not defined";
          throw;
        }
      });
  }

  EcalDQMonitor::~EcalDQMonitor()
  {
    if(verbosity_ > 2) edm::LogInfo("EcalDQM") << moduleName_ << ": Deleting workers";

    executeOnWorkers_([](DQWorker* worker){
        delete worker;
      }, "Dtor");
  }

  /*static*/
  void
  EcalDQMonitor::fillDescriptions(edm::ParameterSetDescription& _desc)
  {
    _desc.addUntracked<std::string>("moduleName", "Ecal Monitor Module");
    _desc.addUntracked<std::vector<std::string> >("workers");
    _desc.addUntracked<int>("verbosity", 0);

    edm::ParameterSetDescription commonParameters;
    commonParameters.addUntracked<bool>("onlineMode", false);
    commonParameters.addUntracked<bool>("willConvertToEDM", true);
    _desc.addUntracked("commonParameters", commonParameters);
  }

  void
  EcalDQMonitor::ecaldqmGetSetupObjects(edm::EventSetup const& _es)
  {
    if(!checkElectronicsMap(false)){
      // set up electronicsMap in EcalDQMCommonUtils
      edm::ESHandle<EcalElectronicsMapping> elecMapHandle;
      _es.get<EcalMappingRcd>().get(elecMapHandle);
      setElectronicsMap(elecMapHandle.product());
    }

    if(!checkTrigTowerMap(false)){
      // set up trigTowerMap in EcalDQMCommonUtils
      edm::ESHandle<EcalTrigTowerConstituentsMap> ttMapHandle;
      _es.get<IdealGeometryRecord>().get(ttMapHandle);
      setTrigTowerMap(ttMapHandle.product());
    }

    if(!checkGeometry(false)){
      edm::ESHandle<CaloGeometry> geomHandle;
      _es.get<CaloGeometryRecord>().get(geomHandle);
      setGeometry(geomHandle.product());
    }

    if(!checkTopology(false)){
      // set up trigTowerMap in EcalDQMCommonUtils
      edm::ESHandle<CaloTopology> topoHandle;
      _es.get<CaloTopologyRecord>().get(topoHandle);
      setTopology(topoHandle.product());
    }
  }

  void
  EcalDQMonitor::ecaldqmReleaseHistograms()
  {
    executeOnWorkers_([](DQWorker* worker){
        worker->releaseMEs();
      }, "releaseMEs", "releasing histograms");
  }

  void
  EcalDQMonitor::ecaldqmBeginRun(edm::Run const& _run, edm::EventSetup const& _es)
  {
    executeOnWorkers_([&_run, &_es](DQWorker* worker){
        if(worker->onlineMode()) worker->setTime(time(0));
        worker->setRunNumber(_run.run());
        worker->beginRun(_run, _es);
      }, "beginRun");

    if(verbosity_ > 0) edm::LogInfo("EcalDQM") << moduleName_ << "::ecaldqmBeginRun";
  }

  void
  EcalDQMonitor::ecaldqmEndRun(edm::Run const& _run, edm::EventSetup const& _es)
  {
    executeOnWorkers_([&_run, &_es](DQWorker* worker){
        if(worker->onlineMode()) worker->setTime(time(0));
        worker->setRunNumber(_run.run());
        worker->endRun(_run, _es);
      }, "endRun");

    if(verbosity_ > 0) edm::LogInfo("EcalDQM") << moduleName_ << "::ecaldqmEndRun";
  }

  void
  EcalDQMonitor::ecaldqmBeginLuminosityBlock(edm::LuminosityBlock const& _lumi, edm::EventSetup const& _es)
  {
    executeOnWorkers_([&_lumi, &_es](DQWorker* worker){
        if(worker->onlineMode()) worker->setTime(time(0));
        worker->setLumiNumber(_lumi.luminosityBlock());
        worker->beginLuminosityBlock(_lumi, _es);
      }, "beginLumi");

    if(verbosity_ > 1) edm::LogInfo("EcalDQM") << moduleName_ << "::ecaldqmBeginLuminosityBlock";
  }

  void
  EcalDQMonitor::ecaldqmEndLuminosityBlock(edm::LuminosityBlock const& _lumi, edm::EventSetup const& _es)
  {
    executeOnWorkers_([&_lumi, &_es](DQWorker* worker){
        if(worker->onlineMode()) worker->setTime(time(0));
        worker->setLumiNumber(_lumi.luminosityBlock());
        worker->endLuminosityBlock(_lumi, _es);
      }, "endLumi");

    if(verbosity_ > 2) edm::LogInfo("EcalDQM") << moduleName_ << "::ecaldqmEndLuminosityBlock";
  }
}
