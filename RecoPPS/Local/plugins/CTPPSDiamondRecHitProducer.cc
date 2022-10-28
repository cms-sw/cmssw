/****************************************************************************
 *
 * This is a part of PPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *
 ****************************************************************************/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"

#include "RecoPPS/Local/interface/CTPPSDiamondRecHitProducerAlgorithm.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationRcd.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationLUTRcd.h"

class CTPPSDiamondRecHitProducer : public edm::stream::EDProducer<> {
public:
  explicit CTPPSDiamondRecHitProducer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondDigi> > digiToken_;
  edm::ESGetToken<PPSTimingCalibration, PPSTimingCalibrationRcd> timingCalibrationToken_;
  edm::ESGetToken<PPSTimingCalibrationLUT, PPSTimingCalibrationLUTRcd> timingCalibrationLUTToken_;
  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> geometryToken_;

  /// A watcher to detect timing calibration changes.
  edm::ESWatcher<PPSTimingCalibrationRcd> calibWatcher_;

  bool applyCalib_;
  CTPPSDiamondRecHitProducerAlgorithm algo_;
};

CTPPSDiamondRecHitProducer::CTPPSDiamondRecHitProducer(const edm::ParameterSet& iConfig)
    : digiToken_(consumes<edm::DetSetVector<CTPPSDiamondDigi> >(iConfig.getParameter<edm::InputTag>("digiTag"))),
      geometryToken_(esConsumes<CTPPSGeometry, VeryForwardRealGeometryRecord>()),
      applyCalib_(iConfig.getParameter<bool>("applyCalibration")),
      algo_(iConfig) {
  if (applyCalib_) {
    timingCalibrationToken_ = esConsumes<PPSTimingCalibration, PPSTimingCalibrationRcd>(
        edm::ESInputTag(iConfig.getParameter<std::string>("timingCalibrationTag")));
    timingCalibrationLUTToken_ = esConsumes<PPSTimingCalibrationLUT, PPSTimingCalibrationLUTRcd>();
  }
  produces<edm::DetSetVector<CTPPSDiamondRecHit> >();
}

void CTPPSDiamondRecHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto pOut = std::make_unique<edm::DetSetVector<CTPPSDiamondRecHit> >();

  // get the digi collection
  const auto& digis = iEvent.get(digiToken_);

  if (!digis.empty()) {
    if (applyCalib_ && calibWatcher_.check(iSetup))
      algo_.setCalibration(iSetup.getData(timingCalibrationToken_), iSetup.getData(timingCalibrationLUTToken_));

    // produce the rechits collection
    algo_.build(iSetup.getData(geometryToken_), digis, *pOut);
  }

  iEvent.put(std::move(pOut));
}

void CTPPSDiamondRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descr) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("digiTag", edm::InputTag("ctppsDiamondRawToDigi", "TimingDiamond"))
      ->setComment("input digis collection to retrieve");
  desc.add<std::string>("timingCalibrationTag", "GlobalTag:PPSDiamondTimingCalibration")
      ->setComment("input tag for timing calibrations retrieval");
  desc.add<double>("timeSliceNs", 25.0 / 1024.0)
      ->setComment("conversion constant between HPTDC timing bin size and nanoseconds");
  desc.add<bool>("applyCalibration", true)->setComment("switch on/off the timing calibration");

  descr.add("ctppsDiamondRecHits", desc);
}

DEFINE_FWK_MODULE(CTPPSDiamondRecHitProducer);
