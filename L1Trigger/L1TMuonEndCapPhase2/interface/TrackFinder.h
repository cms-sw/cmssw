#ifndef L1Trigger_L1TMuonEndCapPhase2_TrackFinder_h
#define L1Trigger_L1TMuonEndCapPhase2_TrackFinder_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"

namespace emtf::phase2 {

    class TrackFinder {
        public:
            explicit TrackFinder(
                    const edm::ParameterSet&,
                    edm::ConsumesCollector&&
            );

            ~TrackFinder();

            void process(
                    // Input
                    const edm::Event&,
                    const edm::EventSetup&,
                    // Output
                    EMTFHitCollection&,
                    EMTFTrackCollection&,
                    EMTFInputCollection&
            );

            void on_job_begin();

            void on_job_end();

        private:
            EMTFContext context_;

            std::vector<std::unique_ptr<TPCollector>> tp_collectors_;
            std::vector<std::unique_ptr<SectorProcessor>> sector_processors_;
    };
}

#endif
