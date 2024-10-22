#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLTHost.h"
#include "CalibTracker/Records/interface/SiPixelGainCalibrationForHLTSoARcd.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationForHLTRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <memory>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class SiPixelGainCalibrationForHLTSoAESProducer : public ESProducer {
  public:
    explicit SiPixelGainCalibrationForHLTSoAESProducer(const edm::ParameterSet& iConfig);
    std::unique_ptr<SiPixelGainCalibrationForHLTHost> produce(const SiPixelGainCalibrationForHLTSoARcd& iRecord);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::ESGetToken<SiPixelGainCalibrationForHLT, SiPixelGainCalibrationForHLTRcd> gainsToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
  };

  SiPixelGainCalibrationForHLTSoAESProducer::SiPixelGainCalibrationForHLTSoAESProducer(const edm::ParameterSet& iConfig)
      : ESProducer(iConfig) {
    auto cc = setWhatProduced(this);
    gainsToken_ = cc.consumes();
    geometryToken_ = cc.consumes();
  }

  void SiPixelGainCalibrationForHLTSoAESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    descriptions.addWithDefaultLabel(desc);
  }

  std::unique_ptr<SiPixelGainCalibrationForHLTHost> SiPixelGainCalibrationForHLTSoAESProducer::produce(
      const SiPixelGainCalibrationForHLTSoARcd& iRecord) {
    auto const& gains = iRecord.get(gainsToken_);
    auto const& geom = iRecord.get(geometryToken_);

    auto product = std::make_unique<SiPixelGainCalibrationForHLTHost>(gains.data().size(), cms::alpakatools::host());

    // bizzarre logic (looking for fist strip-det) don't ask
    auto const& dus = geom.detUnits();
    unsigned int n_detectors = dus.size();
    for (unsigned int i = 1; i < 7; ++i) {
      const auto offset = geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
      if (offset != dus.size() && dus[offset]->type().isTrackerStrip()) {
        if (n_detectors > offset)
          n_detectors = offset;
      }
    }

    LogDebug("SiPixelGainCalibrationForHLTSoA")
        << "caching calibs for " << n_detectors << " pixel detectors of size " << gains.data().size() << '\n'
        << "sizes " << sizeof(char) << ' ' << sizeof(uint8_t) << ' ' << sizeof(siPixelGainsSoA::DecodingStructure);

    for (size_t i = 0; i < gains.data().size(); i = i + 2) {
      product->view().v_pedestals()[i / 2].gain = gains.data()[i];
      product->view().v_pedestals()[i / 2].ped = gains.data()[i + 1];
    }

    //std::copy here
    // do not read back from the (possibly write-combined) memory buffer
    auto minPed = gains.getPedLow();
    auto maxPed = gains.getPedHigh();
    auto minGain = gains.getGainLow();
    auto maxGain = gains.getGainHigh();
    auto nBinsToUseForEncoding = 253;

    // we will simplify later (not everything is needed....)
    product->view().minPed() = minPed;
    product->view().maxPed() = maxPed;
    product->view().minGain() = minGain;
    product->view().maxGain() = maxGain;

    product->view().numberOfRowsAveragedOver() = 80;
    product->view().nBinsToUseForEncoding() = nBinsToUseForEncoding;
    product->view().deadFlag() = 255;
    product->view().noisyFlag() = 254;

    product->view().pedPrecision() = static_cast<float>(maxPed - minPed) / nBinsToUseForEncoding;
    product->view().gainPrecision() = static_cast<float>(maxGain - minGain) / nBinsToUseForEncoding;

    LogDebug("SiPixelGainCalibrationForHLTSoA")
        << "precisions g " << product->view().pedPrecision() << ' ' << product->view().gainPrecision();

    // fill the index map
    auto const& ind = gains.getIndexes();
    LogDebug("SiPixelGainCalibrationForHLTSoA") << ind.size() << " " << n_detectors;

    for (auto i = 0U; i < n_detectors; ++i) {
      auto p = std::lower_bound(
          ind.begin(), ind.end(), dus[i]->geographicalId().rawId(), SiPixelGainCalibrationForHLT::StrictWeakOrdering());
      assert(p != ind.end() && p->detid == dus[i]->geographicalId());
      assert(p->iend <= gains.data().size());
      assert(p->iend >= p->ibegin);
      assert(0 == p->ibegin % 2);
      assert(0 == p->iend % 2);
      assert(p->ibegin != p->iend);
      assert(p->ncols > 0);

      product->view().modStarts()[i] = p->ibegin;
      product->view().modEnds()[i] = p->iend;
      product->view().modCols()[i] = p->ncols;

      if (ind[i].detid != dus[i]->geographicalId())
        LogDebug("SiPixelGainCalibrationForHLTSoA") << ind[i].detid << "!=" << dus[i]->geographicalId();
    }

    return product;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(SiPixelGainCalibrationForHLTSoAESProducer);
