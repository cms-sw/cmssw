#ifndef DataFormats_L1TMuonPhase2_EMTFInput_h
#define DataFormats_L1TMuonPhase2_EMTFInput_h

#include <cstdint>
#include <vector>

#include "DataFormats/L1TMuon/interface/L1TMuonSubsystems.h"

namespace l1t::phase2 {

    class EMTFInput {
        public:
            typedef std::vector<uint16_t> hits_t;
            typedef std::vector<uint16_t> segs_t;


            EMTFInput(): 
                endcap_(0),
                sector_(0),
                bx_(0),
                hits_{},
                segs_{}
            {
                // Do Nothing
            }

            ~EMTFInput() {
                // Do Nothing
            }

            // Setters
            void setEndcap(int16_t aEndcap) { endcap_ = aEndcap; }
            void setSector(int16_t aSector) { sector_ = aSector; }
            void setBx(int16_t aBx) { bx_ = aBx; }
            void setHits(const hits_t& aHits) { hits_ = aHits; }
            void setSegs(const segs_t& aSegs) { segs_ = aSegs; }

            // Getters
            int16_t endcap() const { return endcap_; }
            int16_t sector() const { return sector_; }
            int16_t bx() const { return bx_; }
            const hits_t& hits() const { return hits_; }
            const segs_t& segs() const { return segs_; }

        private:
            int16_t endcap_;
            int16_t sector_;
            int16_t bx_;
            hits_t hits_;
            segs_t segs_;
    };

    typedef std::vector<EMTFInput> EMTFInputCollection;

}  // namespace l1t::phase2

#endif  // DataFormats_L1TMuonPhase2_EMTFInput_h not defined
