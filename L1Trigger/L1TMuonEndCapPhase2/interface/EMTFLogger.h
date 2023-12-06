
#ifndef L1Trigger_L1TMuonEndCapPhase2_EMTFLoggger
#define L1Trigger_L1TMuonEndCapPhase2_EMTFLoggger

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"

namespace emtf::phase2 {

    class EMTFLoggger {
        public:
            EMTFLoggger(const EMTFContext&);

            ~EMTFLoggger();

            // Sections
            void print_section_header(const std:string&);

            void print_section_footer(const std:string&);

            void print_subsection_header(const std:string&);

            void print_subsection_footer(const std:string&);

            // Data
            void print_segment(const int& lvl, const segment_t&);
            void print_track(const int& lvl, const track_t&);
            void print_track_features(const int& lvl, const track_t::features_t&);

        private:
            const EMTFContext& context_;

    };
}

#endif // L1Trigger_L1TMuonEndCapPhase2_EMTFLoggger
