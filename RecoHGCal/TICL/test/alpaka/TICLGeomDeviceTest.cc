#include <string>

#include "CondFormats/HGCalObjects/interface/alpaka/TICLGeomDevice.h"
#include "CondFormats/HGCalObjects/interface/alpaka/TICLGeomLookupDevice.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "TICLGeomDeviceCheck.h"

// Exercises the TICLGeom device collection: runs a kernel on the alpaka
// backend that looks up every cell by its rawDetId with ticlgeom::indexOf
// and counts mismatches. Proves the automatic host to device ES transfer.
namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TICLGeomDeviceTest : public stream::EDProducer<> {
  public:
    explicit TICLGeomDeviceTest(const edm::ParameterSet& iConfig)
        : EDProducer(iConfig),
          ticlGeomToken_(esConsumes(iConfig.getParameter<edm::ESInputTag>("src"))),
          ticlGeomLookupToken_(esConsumes(iConfig.getParameter<edm::ESInputTag>("src"))) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::ESInputTag>("src", edm::ESInputTag())->setComment("Label of the TICLGeom device collection");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event& iEvent, device::EventSetup const& iSetup) override {
      auto const& ticlGeom = iSetup.getData(ticlGeomToken_);
      auto const& ticlGeomLookup = iSetup.getData(ticlGeomLookupToken_);

      auto const common = ticlGeom.const_view().common();
      const int32_t n = common.metadata().size();
      const int32_t nBad = ticlgeomtest::checkLookup(iEvent.queue(), common, ticlGeomLookup.const_view());

      if (nBad != 0) {
        throw cms::Exception("TICLGeomDeviceLookup")
            << nBad << " of " << n << " cells failed the device-side indexOf round trip";
      }
      edm::LogPrint("TICLGeomDeviceTest") << n << " cells verified on the device via indexOf and denseIdOf ("
                                          << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) << ")";
    }

  private:
    device::ESGetToken<TICLGeomDevice, CaloGeometryRecord> ticlGeomToken_;
    device::ESGetToken<TICLGeomLookupDevice, CaloGeometryRecord> ticlGeomLookupToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TICLGeomDeviceTest);
