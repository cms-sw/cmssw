#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/TrackFinder.h"

#include "L1TMuonEndCapPhase2TrackProducer.h"

L1TMuonEndCapPhase2TrackProducer::L1TMuonEndCapPhase2TrackProducer(
        const edm::ParameterSet& pset
):
    track_finder_(std::make_unique<emtf::phase2::TrackFinder>(pset, consumesCollector()))
{
    hit_token_ = produces<emtf::phase2::EMTFHitCollection>();
    trk_token_ = produces<emtf::phase2::EMTFTrackCollection>();
    in_token_  = produces<emtf::phase2::EMTFInputCollection>();
}

L1TMuonEndCapPhase2TrackProducer::~L1TMuonEndCapPhase2TrackProducer() {
    // Do nothing
}

void L1TMuonEndCapPhase2TrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setUnknown();

    descriptions.addDefault(desc);
}

void L1TMuonEndCapPhase2TrackProducer::produce(edm::Event& event, const edm::EventSetup& event_setup) {
    emtf::phase2::EMTFHitCollection out_hits;
    emtf::phase2::EMTFTrackCollection out_tracks;
    emtf::phase2::EMTFInputCollection out_inputs;

    // Forward event to track finder 
    track_finder_->process(
            event, event_setup, 
            out_hits, out_tracks, out_inputs
    );

    // Output
    event.emplace(hit_token_, std::move(out_hits));
    event.emplace(trk_token_, std::move(out_tracks));
    event.emplace(in_token_ , std::move(out_inputs));
}

void L1TMuonEndCapPhase2TrackProducer::beginStream(edm::StreamID stream_id) {
    track_finder_->on_job_begin();
}

void L1TMuonEndCapPhase2TrackProducer::endStream() {
    track_finder_->on_job_end();
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonEndCapPhase2TrackProducer);
