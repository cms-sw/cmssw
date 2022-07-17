#ifndef DQWorker_H
#define DQWorker_H

#include <map>
#include <string>
#include <vector>

#include "DQM/EcalCommon/interface/MESet.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "oneapi/tbb/concurrent_unordered_map.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
namespace edm {
  class Run;
  class LuminosityBlock;
  class Event;
  class EventSetup;
  class ParameterSet;
  class ParameterSetDescription;
  class ConsumesCollector;
}  // namespace edm

namespace ecaldqm {

  class WorkerFactoryStore;

  class DQWorker {
    friend class WorkerFactoryStore;

  private:
    struct Timestamp {
      time_t now;
      edm::RunNumber_t iRun;
      edm::LuminosityBlockNumber_t iLumi;
      edm::EventNumber_t iEvt;
      Timestamp() : now(0), iRun(0), iLumi(0), iEvt(0) {}
    };

  protected:
    typedef dqm::legacy::DQMStore DQMStore;
    typedef dqm::legacy::MonitorElement MonitorElement;

    void setVerbosity(int _verbosity) { verbosity_ = _verbosity; }
    void initialize(std::string const &_name, edm::ParameterSet const &);

    virtual void setME(edm::ParameterSet const &);
    virtual void setSource(edm::ParameterSet const &) {}  // for clients
    virtual void setParams(edm::ParameterSet const &) {}

  public:
    DQWorker();
    virtual ~DQWorker() noexcept(false);

    static void fillDescriptions(edm::ParameterSetDescription &_desc);
    void setTokens(edm::ConsumesCollector &);

    virtual void beginRun(edm::Run const &, edm::EventSetup const &) {}
    virtual void endRun(edm::Run const &, edm::EventSetup const &) {}

    virtual void beginLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}
    virtual void endLuminosityBlock(edm::LuminosityBlock const &, edm::EventSetup const &) {}

    virtual void bookMEs(DQMStore::IBooker &);
    virtual void releaseMEs();

    // old ecaldqmGetSetupObjects (old global vars)
    // These are objects obtained from EventSetup and stored
    // inside each module (which inherit from DQWorker).
    // Before, EcalCommon functions could access these through
    // global functions, but now we need to pass them from the
    // modules to functions in EcalCommon, such as in
    // EcalDQMCommonUtils, MESetBinningUtils, all MESets, etc.
    //
    // The global variables were removed as they were against
    // CMSSW rules, and potentially led to undefined behavior
    // (data race) at IOV boundaries. They also relied on a mutex
    // which leads to poor multi-threading performance.
    // Original issue here:
    // https://github.com/cms-sw/cmssw/issues/28858

    void setSetupObjects(edm::EventSetup const &);
    void setSetupObjectsEndLumi(edm::EventSetup const &);

    bool checkElectronicsMap(bool = true);
    bool checkTrigTowerMap(bool = true);
    bool checkGeometry(bool = true);
    bool checkTopology(bool = true);

    EcalElectronicsMapping const *GetElectronicsMap();
    EcalTrigTowerConstituentsMap const *GetTrigTowerMap();
    CaloGeometry const *GetGeometry();
    CaloTopology const *GetTopology();
    EcalDQMSetupObjects const getEcalDQMSetupObjects();

    edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> elecMapHandle;
    edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> ttMapHandle;
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomHandle;
    edm::ESGetToken<CaloTopology, CaloTopologyRecord> topoHandle;

    edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> elecMapHandleEndLumi;
    edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> ttMapHandleEndLumi;
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomHandleEndLumi;
    edm::ESGetToken<CaloTopology, CaloTopologyRecord> topoHandleEndLumi;

    void setTime(time_t _t) { timestamp_.now = _t; }
    void setRunNumber(edm::RunNumber_t _r) { timestamp_.iRun = _r; }
    void setLumiNumber(edm::LuminosityBlockNumber_t _l) { timestamp_.iLumi = _l; }
    void setEventNumber(edm::EventNumber_t _e) { timestamp_.iEvt = _e; }

    std::string const &getName() const { return name_; }
    bool onlineMode() const { return onlineMode_; }

  protected:
    void print_(std::string const &, int = 0) const;

    std::string name_;
    MESetCollection MEs_;
    bool booked_;

    Timestamp timestamp_;
    int verbosity_;

    // common parameters
    bool onlineMode_;
    bool willConvertToEDM_;

  private:
    EcalDQMSetupObjects edso_;
  };

  typedef DQWorker *(*WorkerFactory)();

  // to be instantiated after the implementation of each worker module
  class WorkerFactoryStore {
  public:
    template <typename Worker>
    struct Registration {
      Registration(std::string const &_name) {
        WorkerFactoryStore::singleton()->registerFactory(_name, []() -> DQWorker * { return new Worker(); });
      }
    };

    void registerFactory(std::string const &_name, WorkerFactory _f) { workerFactories_[_name] = _f; }
    DQWorker *getWorker(std::string const &, int, edm::ParameterSet const &, edm::ParameterSet const &) const;

    static WorkerFactoryStore *singleton();

  private:
    tbb::concurrent_unordered_map<std::string, WorkerFactory> workerFactories_;
  };

}  // namespace ecaldqm

#define DEFINE_ECALDQM_WORKER(TYPE) WorkerFactoryStore::Registration<TYPE> ecaldqm##TYPE##Registration(#TYPE)

#endif
