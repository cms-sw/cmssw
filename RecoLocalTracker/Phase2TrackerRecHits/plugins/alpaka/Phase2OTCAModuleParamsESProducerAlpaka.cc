#include <cstdint>
#include <optional>

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelPhase2Topology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2OTCAModuleParamsHost.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEModuleParams.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/alpaka/Phase2OTCAModuleParamsCollection.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"

#include "Phase2OTCAModules.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // The Phase2StripCPE parameters, copied from the CPE itself, and the surfaces of the OT modules of the CA
  class Phase2OTCAModuleParamsESProducerAlpaka : public ESProducer {
  public:
    explicit Phase2OTCAModuleParamsESProducerAlpaka(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      cpeToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("Phase2StripCPE"));
      geometryToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::ESInputTag>("Phase2StripCPE", edm::ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"))
          ->setComment("the Phase2StripCPE whose parameters are copied");
      descriptions.addWithDefaultLabel(desc);
    }

    std::optional<Phase2OTCAModuleParamsHost> produce(TkPhase2OTCPERecord const& iRecord) {
      auto const* cpe = dynamic_cast<Phase2StripCPE const*>(&iRecord.get(cpeToken_));
      if (cpe == nullptr)
        throw cms::Exception("Configuration")
            << "Phase2OTCAModuleParamsESProducerAlpaka: the CPE is not a Phase2StripCPE";
      auto const modules = phase2OTCAHits::caModules(iRecord.get(geometryToken_));

      Phase2OTCAModuleParamsHost product(cms::alpakatools::host(), modules.size());
      auto view = product.view();
      for (uint32_t index = 0; index < modules.size(); ++index) {
        auto const& det = static_cast<PixelGeomDetUnit const&>(*modules[index]);
        // the module's own topology, not the proxy that Phase2StripCPE::Param::topology points to
        auto const* topology =
            dynamic_cast<RectangularPixelPhase2Topology const*>(&det.specificType().specificTopology());
        if (topology == nullptr)
          throw cms::Exception("BadGeometry")
              << "Phase2OTCAModuleParamsESProducerAlpaka: module " << det.geographicalId().rawId()
              << " does not have a RectangularPixelPhase2Topology";
        auto const& param = cpe->moduleParam(det.index());
        auto const& surface = det.surface();
        auto module = view[index];
        module.cpe() = phase2StripCPE::moduleParams(*topology, param.coveredStrips);
        module.xerrLocal() = param.localErr.xx();
        module.yerrLocal() = param.localErr.yy();
        module.frame() = SOAFrame<float>(surface.position().x(),
                                         surface.position().y(),
                                         surface.position().z(),
                                         SOARotation<float>(surface.rotation()));
      }
      return product;
    }

  private:
    edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord> cpeToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(Phase2OTCAModuleParamsESProducerAlpaka);
