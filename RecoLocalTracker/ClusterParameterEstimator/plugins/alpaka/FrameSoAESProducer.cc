#include <memory>
#include <string>
#include <alpaka/alpaka.hpp>
#include <type_traits>

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/CommonTopologies/interface/TrackerGeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/SimpleSeedingLayersTopology.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/Topology.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "RecoLocalTracker/Records/interface/FrameSoARecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/alpaka/FrameSoACollection.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/FrameSoAHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename TrackerTraits>
  class FrameSoAESProducer : public ESProducer {
    using Rotation = SOARotation<float>;
    using Frame = SOAFrame<float>;

  public:
    FrameSoAESProducer(edm::ParameterSet const& iConfig);
    std::unique_ptr<FrameSoAHost> produce(const FrameSoARecord& iRecord);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometry_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topology_;
  };

  using namespace edm;

  template <typename TrackerTraits>
  FrameSoAESProducer<TrackerTraits>::FrameSoAESProducer(const edm::ParameterSet& p) : ESProducer(p) {
    auto const& myname = p.getParameter<std::string>("ComponentName");

    auto cc = setWhatProduced(this, myname);
    geometry_ = cc.consumes();
    topology_ = cc.consumes();
  }

  template <typename TrackerTraits>
  std::unique_ptr<FrameSoAHost> FrameSoAESProducer<TrackerTraits>::produce(const FrameSoARecord& iRecord) {
    const TrackerGeometry* geometry = &iRecord.get(geometry_);
    const TrackerTopology* topology = &iRecord.get(topology_);

    auto const& detUnits = geometry->detUnits();

    auto product = std::make_unique<FrameSoAHost>(TrackerTraits::numberOfModules, cms::alpakatools::host());

    if constexpr (std::is_same_v<TrackerTraits, pixelTopology::Phase1Strip>) {
      int i = 0;
      for (auto layer : phase1PixelStripTopology::layerData) {
        auto step = layer.isStrip2D ? 2 : 1;
        for (auto j = layer.start; j != layer.end; j += step) {
          auto& s = layer.isStrip2D ? geometry->idToDet(topology->glued(detUnits[i]->geographicalId()))->surface()
                                    : detUnits[j]->surface();
          product->view()[i].detFrame() = Frame(s.position().x(), s.position().y(), s.position().z(), s.rotation());
          ++i;
        }
      }
    } else {
      constexpr auto n_detectors =
          TrackerTraits::numberOfModules;  // converting only up to the modules used in the CA topology

      assert(n_detectors <
             detUnits.size());  //still there shouldn't be more modules than what we have from the TrackerGeometry

      for (unsigned i = 0; i != n_detectors; ++i) {
        auto det = detUnits[i];
        auto vv = det->surface().position();
        auto rr = Rotation(det->surface().rotation());
        product->view()[i].detFrame() = Frame(vv.x(), vv.y(), vv.z(), rr);
      }
    }

    return product;
  }

  template <typename TrackerTraits>
  void FrameSoAESProducer<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    std::string name = "FrameSoAPhase1";
    name += TrackerTraits::nameModifier;
    desc.add<std::string>("ComponentName", name);

    descriptions.addWithDefaultLabel(desc);
  }

  using FrameSoAESProducerPhase1 = FrameSoAESProducer<pixelTopology::Phase1>;
  using FrameSoAESProducerPhase2 = FrameSoAESProducer<pixelTopology::Phase2>;
  using FrameSoAESProducerHIonPhase1 = FrameSoAESProducer<pixelTopology::HIonPhase1>;
  using FrameSoAESProducerPhase1Strip = FrameSoAESProducer<pixelTopology::Phase1Strip>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(FrameSoAESProducerPhase1);
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(FrameSoAESProducerPhase1Strip);
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(FrameSoAESProducerPhase2);
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(FrameSoAESProducerHIonPhase1);
