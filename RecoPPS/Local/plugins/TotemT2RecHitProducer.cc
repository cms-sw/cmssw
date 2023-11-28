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

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/CTPPSDetId/interface/TotemT2DetId.h"
#include "DataFormats/TotemReco/interface/TotemT2Digi.h"
#include "DataFormats/TotemReco/interface/TotemT2RecHit.h"

#include "RecoPPS/Local/interface/TotemT2RecHitProducerAlgorithm.h"

#include "Geometry/Records/interface/TotemGeometryRcd.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

class TotemT2RecHitProducer : public edm::stream::EDProducer<> {
public:
  explicit TotemT2RecHitProducer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<edmNew::DetSetVector<TotemT2Digi> > digiToken_;
  edm::ESGetToken<TotemGeometry, TotemGeometryRcd> geometryToken_;
  /// A watcher to detect timing calibration changes.

  const bool applyCalib_;  //Diamond calibration not used
  TotemT2RecHitProducerAlgorithm algo_;
};

TotemT2RecHitProducer::TotemT2RecHitProducer(const edm::ParameterSet& iConfig)
    : digiToken_(consumes<edmNew::DetSetVector<TotemT2Digi> >(iConfig.getParameter<edm::InputTag>("digiTag"))),
      geometryToken_(esConsumes<TotemGeometry, TotemGeometryRcd>()),
      applyCalib_(iConfig.getParameter<bool>("applyCalibration")),
      algo_(iConfig) {
  produces<edmNew::DetSetVector<TotemT2RecHit> >();
}

void TotemT2RecHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto pOut = std::make_unique<edmNew::DetSetVector<TotemT2RecHit> >();

  // get the digi collection
  const auto& digis = iEvent.get(digiToken_);

  if (!digis.empty()) {
    // produce the rechits collection
    algo_.build(iSetup.getData(geometryToken_), digis, *pOut);
  }

  iEvent.put(std::move(pOut));
}

void TotemT2RecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descr) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("digiTag", edm::InputTag("totemT2Digis", "TotemT2"))
      ->setComment("input digis collection to retrieve");
  desc.add<double>("timeSliceNs", 25.0 / 4.0)
      ->setComment("conversion constant between timing bin size and nanoseconds");
  desc.add<bool>("applyCalibration", false)->setComment("switch on/off the timing calibration (not in use)");

  descr.add("totemT2RecHits", desc);
}

DEFINE_FWK_MODULE(TotemT2RecHitProducer);
