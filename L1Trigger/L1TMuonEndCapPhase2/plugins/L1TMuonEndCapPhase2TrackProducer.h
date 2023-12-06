#ifndef L1Trigger_L1TMuonEndCapPhase2_L1TMuonEndCapPhase2TrackProducer_h
#define L1Trigger_L1TMuonEndCapPhase2_L1TMuonEndCapPhase2TrackProducer_h

#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"

class L1TMuonEndCapPhase2TrackProducer : public edm::stream::EDProducer<> {
    public:
        explicit L1TMuonEndCapPhase2TrackProducer(
                const edm::ParameterSet&
        );

        ~L1TMuonEndCapPhase2TrackProducer() override;

        static void fillDescriptions(
                edm::ConfigurationDescriptions&
        );

    private:
        std::unique_ptr<emtf::phase2::TrackFinder> track_finder_;

        edm::EDPutTokenT<emtf::phase2::EMTFHitCollection> hit_token_;
        edm::EDPutTokenT<emtf::phase2::EMTFTrackCollection> trk_token_;
        edm::EDPutTokenT<emtf::phase2::EMTFInputCollection> in_token_;

        // Producer Functions
        void produce(edm::Event&, const edm::EventSetup&) override;
        void beginStream(edm::StreamID) override;
        void endStream() override;
        // void beginRun(edm::Run const&, edm::EventSetup const&) override;
        // void endRun(edm::Run const&, edm::EventSetup const&) override;
        // void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
        // void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
};

#endif

