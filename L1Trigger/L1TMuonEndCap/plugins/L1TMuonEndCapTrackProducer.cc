#include "L1TMuonEndCapTrackProducer.h"

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsRcd.h"

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestRcd.h"

L1TMuonEndCapTrackProducer::L1TMuonEndCapTrackProducer(const edm::ParameterSet& iConfig) :
    track_finder_(new TrackFinder(iConfig, consumesCollector())),
    uGMT_converter_(new MicroGMTConverter()),
    config_(iConfig)
{
  // Make output products
  produces<EMTFHitCollection>                ("");      // All CSC LCTs and RPC clusters received by EMTF
  produces<EMTFTrackCollection>              ("");      // All output EMTF tracks, in same format as unpacked data
  produces<l1t::RegionalMuonCandBxCollection>("EMTF");  // EMTF tracks output to uGMT
}

L1TMuonEndCapTrackProducer::~L1TMuonEndCapTrackProducer() {

}

void L1TMuonEndCapTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // Pull configuration from the EventSetup
  edm::ESHandle<L1TMuonEndCapParams> handle;
  iSetup.get<L1TMuonEndcapParamsRcd>().get( handle ) ;
  std::shared_ptr<L1TMuonEndCapParams> params(new L1TMuonEndCapParams(*(handle.product())));
  // with the magic above you can use params->fwVersion to change emulator's behavior

  // Pull pt LUT from the EventSetup
  edm::ESHandle<L1TMuonEndCapForest> handle2;
  iSetup.get<L1TMuonEndCapForestRcd>().get( handle2 ) ;
  std::shared_ptr<L1TMuonEndCapForest> ptLUT(new L1TMuonEndCapForest(*(handle2.product())));
  // at that point we want to re-initialize the track_finder_ object with the newly pulled ptLUT
  track_finder_->resetPtLUT( std::const_pointer_cast<const L1TMuonEndCapForest>(ptLUT) );

  // Create pointers to output products
  auto out_hits   = std::make_unique<EMTFHitCollection>();
  auto out_tracks = std::make_unique<EMTFTrackCollection>();
  auto out_cands  = std::make_unique<l1t::RegionalMuonCandBxCollection>();

  // Main EMTF emulator process, produces tracks from hits in each sector in each event
  track_finder_->process(iEvent, iSetup, *out_hits, *out_tracks);

  // Convert into uGMT format
  uGMT_converter_->convert_all(*out_tracks, *out_cands);

  // Fill the output products
  iEvent.put(std::move(out_hits)  , "");
  iEvent.put(std::move(out_tracks), "");
  iEvent.put(std::move(out_cands) , "EMTF");
}

void L1TMuonEndCapTrackProducer::beginJob() {

}

void L1TMuonEndCapTrackProducer::endJob() {

}

// Fill 'descriptions' with the allowed parameters
void L1TMuonEndCapTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

// Define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonEndCapTrackProducer);
