#include <memory>

#include "CondFormats/HGCalObjects/interface/TICLGeomHost.h"
#include "CondFormats/HGCalObjects/interface/TICLGeomLookupHost.h"
// also brings in the CopyToDevice specialization that registers the
// automatic host to device transfer of the produced collection
#include "CondFormats/HGCalObjects/interface/alpaka/TICLGeomLookupDevice.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Builds the coarse-block lookup table for a TICLGeom collection. The
  // cells product is ordered by rawDetId, so every silicon wafer / scint
  // ring occupies a contiguous run; this records, per coarse index
  // (ticlgeom::coarseSlot), the run's start row and length. One direct
  // table per subdetector, concatenated (EE | HSi | HSc). Empty for a cell
  // collection that has no HGCal subdetectors.
  class TICLGeomLookupESProducer : public ESProducer {
  public:
    explicit TICLGeomLookupESProducer(edm::ParameterSet const& iConfig) : ESProducer(iConfig) {
      auto cc = setWhatProduced(this);
      geomToken_ = cc.consumes(iConfig.getParameter<edm::ESInputTag>("src"));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::ESInputTag>("src", edm::ESInputTag())->setComment("Label of the TICLGeom cell collection");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<TICLGeomLookupHost> produce(CaloGeometryRecord const& iRecord) {
      using namespace ticlgeom::detail;
      auto const& cells = iRecord.get(geomToken_);
      auto const cellsView = cells.const_view().common();

      const int32_t nSlots = 2 * kSiSlots + kScSlots;  // EE | HSi | HSc
      auto product = std::make_unique<TICLGeomLookupHost>(cms::alpakatools::host(), nSlots);
      auto view = product->view();
      view.eeBase() = 0;
      view.hsiBase() = kSiSlots;
      view.hscBase() = 2 * kSiSlots;
      for (int32_t s = 0; s < nSlots; ++s) {
        view[s].blockStart() = -1;
        view[s].blockCount() = 0;
      }

      // cells are rawDetId-ordered, so each coarse block is a contiguous run
      for (int32_t i = 0; i < cellsView.metadata().size(); ++i) {
        const int32_t slot = ticlgeom::coarseSlot(view, cellsView[i].rawDetId());
        if (slot < 0) {
          continue;
        }
        if (view[slot].blockCount() == 0) {
          view[slot].blockStart() = i;
        }
        view[slot].blockCount() = view[slot].blockCount() + 1;
      }

      return product;
    }

  private:
    edm::ESGetToken<TICLGeomHost, CaloGeometryRecord> geomToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(TICLGeomLookupESProducer);
