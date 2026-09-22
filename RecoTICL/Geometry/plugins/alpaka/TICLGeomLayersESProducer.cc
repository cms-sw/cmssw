#include <algorithm>
#include <memory>

#include "CondFormats/HGCalObjects/interface/TICLGeomLayersHost.h"
// also brings in the CopyToDevice specialization that registers the
// automatic host to device transfer of the produced collection
#include "CondFormats/HGCalObjects/interface/alpaka/TICLGeomLayersDevice.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Per-layer table replacing RecHitTools::getPositionLayer, filled by
  // calling it: endcap and HFNose layer z at row |layer|, barrel first-cell
  // x/y at rows 0..lastLayerBarrel.
  class TICLGeomLayersESProducer : public ESProducer {
  public:
    explicit TICLGeomLayersESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      geomToken_ = cc.consumes();
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<TICLGeomLayersHost> produce(CaloGeometryRecord const& iRecord) {
      auto const& geom = iRecord.get(geomToken_);
      hgcal::RecHitTools tools;
      tools.setGeometry(geom);

      const int32_t nLayers = tools.lastLayer();
      const int32_t nNose = tools.lastLayer(true);
      const int32_t nBarrel = tools.lastLayerBarrel();
      const int32_t n = std::max({nLayers, nNose, nBarrel}) + 1;

      auto product = std::make_unique<TICLGeomLayersHost>(cms::alpakatools::host(), n);
      auto view = product->view();
      for (int32_t lay = 0; lay < n; ++lay) {
        view[lay].z() = (lay >= 1 && lay <= nLayers) ? tools.getPositionLayer(lay).z() : 0.f;
        view[lay].noseZ() = (lay >= 1 && lay <= nNose) ? tools.getPositionLayer(lay, true).z() : 0.f;
        if (lay <= nBarrel) {
          const auto pos = tools.getPositionLayer(lay, false, true);
          view[lay].barrelX() = pos.x();
          view[lay].barrelY() = pos.y();
        } else {
          view[lay].barrelX() = 0.f;
          view[lay].barrelY() = 0.f;
        }
      }

      return product;
    }

  private:
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TICLGeomLayersESProducer);
