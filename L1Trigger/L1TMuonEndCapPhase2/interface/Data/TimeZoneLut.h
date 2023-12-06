#ifndef L1Trigger_L1TMuonEndCapPhase2_TimeZoneLut_h
#define L1Trigger_L1TMuonEndCapPhase2_TimeZoneLut_h

#include <map>
#include <tuple>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace emtf::phase2::data {

    class TimeZoneLut {
        // Static
        public:
            bool in_range(const std::pair<int, int>&, const int&) const;

        // Member
        public:
            TimeZoneLut();

            ~TimeZoneLut();

            void update(
                    const edm::Event&,
                    const edm::EventSetup&);

            int get_timezones(const int&, const int&) const;

        private:
            // Key: Host
            // Value: BX Range
            std::map<int, std::pair<int, int>> lut_;
    };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_TimeZoneLut_h
