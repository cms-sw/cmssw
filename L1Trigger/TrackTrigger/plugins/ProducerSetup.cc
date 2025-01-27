#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <memory>

using namespace std;
using namespace edm;

namespace tt {

  /*! \class  tt::ProducerSetup
   *  \brief  Class to produce setup of Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Apr
   */
  class ProducerSetup : public ESProducer {
  public:
    ProducerSetup(const ParameterSet& iConfig);
    ~ProducerSetup() override {}
    unique_ptr<Setup> produce(const SetupRcd& setupRcd);

  private:
    const ParameterSet iConfig_;
    ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> getTokenTrackerGeometry_;
    ESGetToken<TrackerTopology, TrackerTopologyRcd> getTokenTrackerTopology_;
    ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> getTokenCablingMap_;
    ESGetToken<StubAlgorithm, TTStubAlgorithmRecord> getTokenTTStubAlgorithm_;
  };

  ProducerSetup::ProducerSetup(const ParameterSet& iConfig) : iConfig_(iConfig) {
    auto cc = setWhatProduced(this);
    getTokenTrackerGeometry_ = cc.consumes();
    getTokenTrackerTopology_ = cc.consumes();
    getTokenCablingMap_ = cc.consumes();
    getTokenTTStubAlgorithm_ = cc.consumes();
  }

  unique_ptr<Setup> ProducerSetup::produce(const SetupRcd& setupRcd) {
    const TrackerGeometry& trackerGeometry = setupRcd.get(getTokenTrackerGeometry_);
    const TrackerTopology& trackerTopology = setupRcd.get(getTokenTrackerTopology_);
    const TrackerDetToDTCELinkCablingMap& cablingMap = setupRcd.get(getTokenCablingMap_);
    const ESHandle<StubAlgorithm> handleStubAlgorithm = setupRcd.getHandle(getTokenTTStubAlgorithm_);
    const StubAlgorithmOfficial& stubAlgoritm =
        *dynamic_cast<const StubAlgorithmOfficial*>(&setupRcd.get(getTokenTTStubAlgorithm_));
    const ParameterSet& pSetStubAlgorithm = getParameterSet(handleStubAlgorithm.description()->pid_);
    return make_unique<Setup>(iConfig_, trackerGeometry, trackerTopology, cablingMap, stubAlgoritm, pSetStubAlgorithm);
  }
}  // namespace tt

DEFINE_FWK_EVENTSETUP_MODULE(tt::ProducerSetup);
