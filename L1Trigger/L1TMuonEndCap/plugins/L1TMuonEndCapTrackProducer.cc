#include "L1TMuonEndCapTrackProducer.h"

L1TMuonEndCapTrackProducer::L1TMuonEndCapTrackProducer(const edm::ParameterSet& iConfig)
    : track_finder_(new TrackFinder(iConfig, consumesCollector())),
      uGMT_converter_(new MicroGMTConverter()),
      config_(iConfig) {
  // Make output products
  // produces<l1t::CPPFDigiCollection>          ("");      // CPPF Digis emulated by EMTF using RPCDigis from Legacy RPC PAC
  produces<EMTFHitCollection>("");                      // All CSC LCTs and RPC clusters received by EMTF
  produces<EMTFTrackCollection>("");                    // All output EMTF tracks, in same format as unpacked data
  produces<l1t::RegionalMuonCandBxCollection>("EMTF");  // EMTF tracks output to uGMT
}

L1TMuonEndCapTrackProducer::~L1TMuonEndCapTrackProducer() {}

void L1TMuonEndCapTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Create pointers to output products
  // auto out_cppfs   = std::make_unique<l1t::CPPFDigiCollection>();
  auto out_hits_tmp = std::make_unique<EMTFHitCollection>();
  auto out_hits_ZS = std::make_unique<EMTFHitCollection>();
  auto out_tracks = std::make_unique<EMTFTrackCollection>();
  auto out_cands = std::make_unique<l1t::RegionalMuonCandBxCollection>();

  // Main EMTF emulator process, produces tracks from hits in each sector in each event
  track_finder_->process(iEvent, iSetup, *out_hits_tmp, *out_tracks);

  // Apply ZeroSuppression: Only save RPC hits if there is at least one CSC LCT in the sector
  bool has_LCT[12] = {false};
  for (int iSect = 0; iSect < 12; iSect++) {
    for (const auto& h : *out_hits_tmp) {
      if (h.Is_CSC() && h.Sector_idx() == iSect) {
        has_LCT[iSect] = true;
        break;
      }
    }
  }
  for (const auto& h : *out_hits_tmp) {
    if (has_LCT[h.Sector_idx()] || h.Is_RPC() == 0) {
      out_hits_ZS->push_back(h);
    }
  }

  // // Fill collection of emulated CPPFDigis
  // for (const auto& h : *out_hits_ZS) {
  //   if (h.Is_RPC()) out_cppfs->push_back( h.CreateCPPFDigi() );
  // }

  // Convert into uGMT format
  uGMT_converter_->convert_all(iEvent, *out_tracks, *out_cands);

  // Fill the output products
  // iEvent.put(std::move(out_cppfs),     "");
  iEvent.put(std::move(out_hits_ZS), "");
  iEvent.put(std::move(out_tracks), "");
  iEvent.put(std::move(out_cands), "EMTF");
}

// void L1TMuonEndCapTrackProducer::beginJob() {
// }

// void L1TMuonEndCapTrackProducer::endJob() {
// }

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
