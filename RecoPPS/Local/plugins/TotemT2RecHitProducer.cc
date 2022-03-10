/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
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

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "DataFormats/CTPPSDetId/interface/TotemT2DetId.h"
#include "DataFormats/TotemReco/interface/TotemT2Digi.h"
#include "DataFormats/TotemReco/interface/TotemT2RecHit.h"

#include "RecoPPS/Local/interface/TotemT2RecHitProducerAlgorithm.h"

#include "Geometry/Records/interface/TotemGeometryRcd.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationRcd.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationLUTRcd.h"

class TotemT2RecHitProducer : public edm::stream::EDProducer<> {
public:
  explicit TotemT2RecHitProducer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<edm::DetSetVector<TotemT2Digi> > digiToken_;
  edm::ESGetToken<PPSTimingCalibration, PPSTimingCalibrationRcd> timingCalibrationToken_;
  edm::ESGetToken<PPSTimingCalibrationLUT, PPSTimingCalibrationLUTRcd> timingCalibrationLUTToken_;
  edm::ESGetToken<TotemGeometry, TotemGeometryRcd> geometryToken_;
  /// A watcher to detect timing calibration changes.
  edm::ESWatcher<PPSTimingCalibrationRcd> calibWatcher_;

  const bool applyCalib_;
  TotemT2RecHitProducerAlgorithm algo_;
};

TotemT2RecHitProducer::TotemT2RecHitProducer(const edm::ParameterSet& iConfig)
    : digiToken_(consumes<edm::DetSetVector<TotemT2Digi> >(iConfig.getParameter<edm::InputTag>("digiTag"))),
      geometryToken_(esConsumes<TotemGeometry, TotemGeometryRcd>()),
      applyCalib_(iConfig.getParameter<bool>("applyCalibration")),
      algo_(iConfig) {
  if (applyCalib_)
    timingCalibrationToken_ = esConsumes<PPSTimingCalibration, PPSTimingCalibrationRcd>(
        edm::ESInputTag(iConfig.getParameter<std::string>("timingCalibrationTag")));
  produces<edm::DetSetVector<TotemT2RecHit> >();
}

void TotemT2RecHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto pOut = std::make_unique<edm::DetSetVector<TotemT2RecHit> >();

  // get the digi collection
  edm::Handle<edm::DetSetVector<TotemT2Digi> > digis;
  iEvent.getByToken(digiToken_, digis);

  if (!digis->empty()) {
    if (applyCalib_ && calibWatcher_.check(iSetup)) {
      edm::ESHandle<PPSTimingCalibration> hTimingCalib = iSetup.getHandle(timingCalibrationToken_);
      edm::ESHandle<PPSTimingCalibrationLUT> hTimingCalibLUT = iSetup.getHandle(timingCalibrationLUTToken_);
      algo_.setCalibration(*hTimingCalib, *hTimingCalibLUT);
    }
    // get the geometry
    edm::ESHandle<TotemGeometry> geometry = iSetup.getHandle(geometryToken_);

    // produce the rechits collection
    algo_.build(*geometry, *digis, *pOut);
  }

  iEvent.put(std::move(pOut));
}

void TotemT2RecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descr) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("digiTag", edm::InputTag("totemT2Digis", "TotemT2"))
      ->setComment("input digis collection to retrieve");
  desc.add<std::string>("timingCalibrationTag", "GlobalTag:TotemT2TimingCalibration")
      ->setComment("input tag for timing calibrations retrieval");
  desc.add<double>("timeSliceNs", 25.0 / 1024.0)
      ->setComment("conversion constant between timing bin size and nanoseconds");
  desc.add<bool>("applyCalibration", false)->setComment("switch on/off the timing calibration");

  descr.add("totemT2RecHits", desc);
}

DEFINE_FWK_MODULE(TotemT2RecHitProducer);
