#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
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
    ESGetToken<StubAlgorithm, TTStubAlgorithmRecord> getTokenTTStubAlgorithm_;
    ESGetToken<MagneticField, IdealMagneticFieldRecord> getTokenMagneticField_;
    ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> getTokenTrackerGeometry_;
    ESGetToken<TrackerTopology, TrackerTopologyRcd> getTokenTrackerTopology_;
    ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> getTokenCablingMap_;
    ESGetToken<DDCompactView, IdealGeometryRecord> getTokenGeometryConfiguration_;
    ESGetToken<cms::DDCompactView, IdealGeometryRecord> getTokenGeometryConfigurationDD4hep_;
    bool fromDD4hep_;
  };

  ProducerSetup::ProducerSetup(const ParameterSet& iConfig) : iConfig_(iConfig) {
    fromDD4hep_ = iConfig.getParameter<bool>("fromDD4hep");
    auto cc = setWhatProduced(this);
    getTokenTTStubAlgorithm_ = cc.consumes();
    getTokenMagneticField_ = cc.consumes();
    getTokenTrackerGeometry_ = cc.consumes();
    getTokenTrackerTopology_ = cc.consumes();
    getTokenCablingMap_ = cc.consumes();
    if (fromDD4hep_)
      getTokenGeometryConfigurationDD4hep_ = cc.consumes();
    else
      getTokenGeometryConfiguration_ = cc.consumes();
  }

  unique_ptr<Setup> ProducerSetup::produce(const SetupRcd& setupRcd) {
    const MagneticField& magneticField = setupRcd.get(getTokenMagneticField_);
    const TrackerGeometry& trackerGeometry = setupRcd.get(getTokenTrackerGeometry_);
    const TrackerTopology& trackerTopology = setupRcd.get(getTokenTrackerTopology_);
    const TrackerDetToDTCELinkCablingMap& cablingMap = setupRcd.get(getTokenCablingMap_);
    const ESHandle<StubAlgorithm> handleStubAlgorithm = setupRcd.getHandle(getTokenTTStubAlgorithm_);
    const ParameterSetID& pSetIdTTStubAlgorithm = handleStubAlgorithm.description()->pid_;
    const StubAlgorithmOfficial& stubAlgoritm =
        *dynamic_cast<const StubAlgorithmOfficial*>(&setupRcd.get(getTokenTTStubAlgorithm_));
    const ParameterSet& pSetStubAlgorithm = getParameterSet(handleStubAlgorithm.description()->pid_);
    if (fromDD4hep_) {
      const ESHandle<cms::DDCompactView> handleGeometryConfiguration =
          setupRcd.getHandle(getTokenGeometryConfigurationDD4hep_);
      const ParameterSetID& pSetIdGeometryConfiguration = handleGeometryConfiguration.description()->pid_;
      const ParameterSet& pSetGeometryConfiguration = getParameterSet(handleGeometryConfiguration.description()->pid_);
      return make_unique<Setup>(iConfig_,
                                magneticField,
                                trackerGeometry,
                                trackerTopology,
                                cablingMap,
                                stubAlgoritm,
                                pSetStubAlgorithm,
                                pSetGeometryConfiguration,
                                pSetIdTTStubAlgorithm,
                                pSetIdGeometryConfiguration);
    } else {
      const ESHandle<DDCompactView> handleGeometryConfiguration = setupRcd.getHandle(getTokenGeometryConfiguration_);
      const ParameterSetID& pSetIdGeometryConfiguration = handleGeometryConfiguration.description()->pid_;
      const ParameterSet& pSetGeometryConfiguration = getParameterSet(handleGeometryConfiguration.description()->pid_);
      return make_unique<Setup>(iConfig_,
                                magneticField,
                                trackerGeometry,
                                trackerTopology,
                                cablingMap,
                                stubAlgoritm,
                                pSetStubAlgorithm,
                                pSetGeometryConfiguration,
                                pSetIdTTStubAlgorithm,
                                pSetIdGeometryConfiguration);
    }
  }
}  // namespace tt

DEFINE_FWK_EVENTSETUP_MODULE(tt::ProducerSetup);
