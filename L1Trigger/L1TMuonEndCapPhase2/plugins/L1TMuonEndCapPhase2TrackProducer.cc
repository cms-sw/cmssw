#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/TrackFinder.h"

#include "L1TMuonEndCapPhase2TrackProducer.h"

L1TMuonEndCapPhase2TrackProducer::L1TMuonEndCapPhase2TrackProducer(const edm::ParameterSet& pset)
    : track_finder_(std::make_unique<emtf::phase2::TrackFinder>(pset, consumesCollector())),
      hit_token_(produces<emtf::phase2::EMTFHitCollection>()),
      trk_token_(produces<emtf::phase2::EMTFTrackCollection>()),
      in_token_(produces<emtf::phase2::EMTFInputCollection>()) {}

void L1TMuonEndCapPhase2TrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // Neural Network Models
  desc.add<std::string>("PromptGraphPath", "L1Trigger/L1TMuonEndCapPhase2/data/prompt_model.pb");
  desc.add<std::string>("DisplacedGraphPath", "L1Trigger/L1TMuonEndCapPhase2/data/displaced_model.pb");

  // Input Collections
  desc.add<edm::InputTag>("CSCInput", edm::InputTag("simCscTriggerPrimitiveDigisForEMTF"));
  desc.add<edm::InputTag>("RPCInput", edm::InputTag("rpcRecHitsForEMTF"));
  desc.add<edm::InputTag>("GEMInput", edm::InputTag("simMuonGEMPadDigiClusters"));
  desc.add<edm::InputTag>("ME0Input", edm::InputTag("me0TriggerConvertedPseudoDigis"));
  desc.add<edm::InputTag>("GE0Input", edm::InputTag("ge0TriggerConvertedPseudoDigis"));

  // Enable Subdetectors
  desc.add<bool>("CSCEnabled", true);
  desc.add<bool>("RPCEnabled", true);
  desc.add<bool>("GEMEnabled", true);
  desc.add<bool>("ME0Enabled", true);
  desc.add<bool>("GE0Enabled", false);

  // Bunch-Crossing Settings
  desc.add<int>("MinBX", -2);
  desc.add<int>("MaxBX", 2);
  desc.add<int>("BXWindow", 1);

  desc.add<int>("CSCInputBXShift", -8);
  desc.add<int>("RPCInputBXShift", 0);
  desc.add<int>("GEMInputBXShift", 0);
  desc.add<int>("ME0InputBXShift", -8);

  // Primitive Settings
  desc.add<bool>("IncludeNeighborEnabled", true);

  // Debug Utils
  desc.addUntracked<int>("Verbosity", 3);
  desc.add<std::string>("ValidationDirectory", "L1Trigger/L1TMuonEndCapPhase2/data/validation");

  // Register
  descriptions.add("L1TMuonEndCapPhase2TrackProducer", desc);
}

void L1TMuonEndCapPhase2TrackProducer::produce(edm::Event& event, const edm::EventSetup& event_setup) {
  emtf::phase2::EMTFHitCollection out_hits;
  emtf::phase2::EMTFTrackCollection out_tracks;
  emtf::phase2::EMTFInputCollection out_inputs;

  // Forward event to track finder
  track_finder_->process(event, event_setup, out_hits, out_tracks, out_inputs);

  // Output
  event.emplace(hit_token_, std::move(out_hits));
  event.emplace(trk_token_, std::move(out_tracks));
  event.emplace(in_token_, std::move(out_inputs));
}

void L1TMuonEndCapPhase2TrackProducer::beginStream(edm::StreamID stream_id) { track_finder_->onJobBegin(); }

void L1TMuonEndCapPhase2TrackProducer::endStream() { track_finder_->onJobEnd(); }

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonEndCapPhase2TrackProducer);
