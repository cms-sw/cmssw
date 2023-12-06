#ifndef DataFormats_L1TMuonPhase2_EMTFHit_h
#define DataFormats_L1TMuonPhase2_EMTFHit_h

#include <cstdint>
#include <vector>

#include "DataFormats/L1TMuon/interface/L1TMuonSubsystems.h"

namespace l1t::phase2 {

    class EMTFHit {
        public:
            EMTFHit(): 
                id_(0),

                raw_det_id_(0),
                subsystem_(0),
                endcap_(0),
                sector_(0),
                subsector_(0),
                station_(0),
                ring_(0),
                roll_(0),
                layer_(0),
                chamber_(0),

                csc_id_(0),
                csc_fr_(0),

                strip_(0),
                strip_lo_(0),
                strip_hi_(0),
                strip_quart_(0),       // Run 3
                strip_eighth_(0),      // Run 3
                strip_quart_bit_(0),   // Run 3
                strip_eighth_bit_(0),  // Run 3

                wire1_(0),
                wire2_(0),

                bx_(0),
                subbx_(0),

                quality_(0),
                pattern_(0),

                glob_phi_(0),
                glob_theta_(0),
                glob_perp_(0),
                glob_z_(0),
                glob_time_(0),

                emtf_chamber_(0),
                emtf_segment_(0),
                emtf_phi_(0),
                emtf_bend_(0),
                emtf_slope_(0),
                emtf_theta1_(0),
                emtf_theta2_(0),
                emtf_qual1_(0),
                emtf_qual2_(0),
                emtf_time_(0),
                emtf_site_(0),
                emtf_host_(0),
                emtf_zones_(0),
                emtf_timezones_(0),

                flag_neighbor_(false),
                flag_substitute_(false),
                flag_valid_(false) 
            {
                // Do Nothing
            }

            ~EMTFHit() {
                // Do Nothing
            }

            // Setters
            void setId(uint16_t aId) { id_ = aId; }

            void setRawDetId(uint32_t aRawDetId) { raw_det_id_ = aRawDetId; }
            void setSubsystem(int16_t aSubsystem) { subsystem_ = aSubsystem; }
            void setEndcap(int16_t aEndcap) { endcap_ = aEndcap; }
            void setSector(int16_t aSector) { sector_ = aSector; }
            void setSubsector(int16_t aSubsector) { subsector_ = aSubsector; }
            void setStation(int16_t aStation) { station_ = aStation; }
            void setRing(int16_t aRing) { ring_ = aRing; }
            void setRoll(int16_t aRoll) { roll_ = aRoll; }
            void setLayer(int16_t aLayer) { layer_ = aLayer; }
            void setChamber(int16_t aChamber) { chamber_ = aChamber; }

            void setCscId(int16_t aCscid) { csc_id_ = aCscid; }
            void setCscFR(int16_t aCscfr) { csc_fr_ = aCscfr; }

            void setStrip(int16_t aStrip) { strip_ = aStrip; }
            void setStripLo(int16_t aStripLo) { strip_lo_ = aStripLo; }
            void setStripHi(int16_t aStripHi) { strip_hi_ = aStripHi; }
            void setStripQuart(int16_t aStripQuart) { strip_quart_ = aStripQuart; }                     // Run 3
            void setStripEighth(int16_t aStripEighth) { strip_eighth_ = aStripEighth; }                 // Run 3
            void setStripQuartBit(int16_t aStripQuartBit) { strip_quart_bit_ = aStripQuartBit; }        // Run 3
            void setStripEighthBit(int16_t aStripEighthBit) { strip_eighth_bit_ = aStripEighthBit; }    // Run 3

            void setWire1(int16_t aWire1) { wire1_ = aWire1; }
            void setWire2(int16_t aWire2) { wire2_ = aWire2; }

            void setBend(int16_t aBend) { bend_ = aBend; }
            void setSlope(int16_t aSlope) { slope_ = aSlope; }

            void setBx(int16_t aBx) { bx_ = aBx; }
            void setSubbx(int16_t aSubbx) { subbx_ = aSubbx; }

            void setQuality(int16_t aQuality) { quality_ = aQuality; }
            void setPattern(int16_t aPattern) { pattern_ = aPattern; }

            void setGlobPhi(float aGlobPhi) { glob_phi_ = aGlobPhi; }
            void setGlobTheta(float aGlobTheta) { glob_theta_ = aGlobTheta; }
            void setGlobPerp(float aGlobPerp) { glob_perp_ = aGlobPerp; }
            void setGlobZ(float aGlobZ) { glob_z_ = aGlobZ; }
            void setGlobTime(float aGlobTime) { glob_time_ = aGlobTime; }

            void setEmtfChamber(int16_t aEmtfChamber) { emtf_chamber_ = aEmtfChamber; }
            void setEmtfSegment(int16_t aEmtfSegment) { emtf_segment_ = aEmtfSegment; }
            void setEmtfPhi(int16_t aEmtfPhi) { emtf_phi_ = aEmtfPhi; }
            void setEmtfBend(int16_t aEmtfBend) { emtf_bend_ = aEmtfBend; }
            void setEmtfTheta1(int16_t aEmtfTheta1) { emtf_theta1_ = aEmtfTheta1; }
            void setEmtfTheta2(int16_t aEmtfTheta2) { emtf_theta2_ = aEmtfTheta2; }
            void setEmtfQual1(int16_t aEmtfQual1) { emtf_qual1_ = aEmtfQual1; }
            void setEmtfQual2(int16_t aEmtfQual2) { emtf_qual2_ = aEmtfQual2; }
            void setEmtfTime(int16_t aEmtfTime) { emtf_time_ = aEmtfTime; }
            void setEmtfSite(int16_t aEmtfSite) { emtf_site_ = aEmtfSite; }
            void setEmtfHost(int16_t aEmtfHost) { emtf_host_ = aEmtfHost; }
            void setEmtfZones(int16_t aEmtfZones) { emtf_zones_ = aEmtfZones; }
            void setEmtfTimezones(int16_t aEmtfTimezones) { emtf_timezones_ = aEmtfTimezones; }

            void setFlagNeighbor(bool aNeighbor) { flag_neighbor_ = aNeighbor; }
            void setFlagSubstitute(bool aSubstitute) { flag_substitute_ = aSubstitute; }
            void setFlagValid(bool aValid) { flag_valid_ = aValid; }

            // Getters
            uint16_t id() const { return id_; }

            uint32_t rawDetId() const { return raw_det_id_; }
            int16_t subsystem() const { return subsystem_; }
            int16_t endcap() const { return endcap_; }
            int16_t sector() const { return sector_; }
            int16_t subsector() const { return subsector_; }
            int16_t station() const { return station_; }
            int16_t ring() const { return ring_; }
            int16_t roll() const { return roll_; }
            int16_t layer() const { return layer_; }
            int16_t chamber() const { return chamber_; }

            int16_t cscId() const { return csc_id_; }
            int16_t cscFR() const { return csc_fr_; }

            int16_t strip() const { return strip_; }
            int16_t stripLo() const { return strip_lo_; }
            int16_t stripHi() const { return strip_hi_; }
            int16_t stripQuart() const { return strip_quart_; }          // Run 3
            int16_t stripEighth() const { return strip_eighth_; }        // Run 3
            int16_t stripQuartBit() const { return strip_quart_bit_; }   // Run 3
            int16_t stripEighthBit() const { return strip_eighth_bit_; } // Run 3

            int16_t wire1() const { return wire1_; }
            int16_t wire2() const { return wire2_; }

            int16_t bend() const { return bend_; }
            int16_t slope() const { return slope_; }

            int16_t bx() const { return bx_; }
            int16_t subbx() const { return subbx_; }

            int16_t quality() const { return quality_; }
            int16_t pattern() const { return pattern_; }

            float globPhi() const { return glob_phi_; }
            float globTheta() const { return glob_theta_; }
            float globPerp() const { return glob_perp_; }
            float globZ() const { return glob_z_; }
            float globTime() const { return glob_time_; }

            int16_t emtfChamber() const { return emtf_chamber_; }
            int16_t emtfSegment() const { return emtf_segment_; }
            int16_t emtfPhi() const { return emtf_phi_; }
            int16_t emtfBend() const { return emtf_bend_; }
            int16_t emtfTheta1() const { return emtf_theta1_; }
            int16_t emtfTheta2() const { return emtf_theta2_; }
            int16_t emtfQual1() const { return emtf_qual1_; }
            int16_t emtfQual2() const { return emtf_qual2_; }
            int16_t emtfTime() const { return emtf_time_; }
            int16_t emtfSite() const { return emtf_site_; }
            int16_t emtfHost() const { return emtf_host_; }
            int16_t emtfZones() const { return emtf_zones_; }
            int16_t emtfTimezones() const { return emtf_timezones_; }

            bool flagNeighbor() const { return flag_neighbor_; }
            bool flagSubstitute() const { return flag_substitute_; }
            bool flagValid() const { return flag_valid_; }

        private:
            uint16_t id_;

            uint32_t raw_det_id_;
            int16_t subsystem_;
            int16_t endcap_;
            int16_t sector_;
            int16_t subsector_;
            int16_t station_;
            int16_t ring_;
            int16_t roll_;
            int16_t layer_;
            int16_t chamber_;

            int16_t csc_id_;
            int16_t csc_fr_; // front/rear

            int16_t strip_;
            int16_t strip_lo_;
            int16_t strip_hi_;
            int16_t strip_quart_;
            int16_t strip_eighth_;
            int16_t strip_quart_bit_;
            int16_t strip_eighth_bit_;

            int16_t wire1_;
            int16_t wire2_;

            int16_t bend_;
            int16_t slope_;

            int16_t bx_;
            int16_t subbx_;

            int16_t quality_;
            int16_t pattern_;

            float glob_phi_;
            float glob_theta_;
            float glob_perp_;
            float glob_z_;
            float glob_time_;

            int16_t emtf_chamber_;
            int16_t emtf_segment_;
            int16_t emtf_phi_;
            int16_t emtf_bend_;
            int16_t emtf_slope_;
            int16_t emtf_theta1_;
            int16_t emtf_theta2_;
            int16_t emtf_qual1_;
            int16_t emtf_qual2_;
            int16_t emtf_time_;
            int16_t emtf_site_;
            int16_t emtf_host_;
            int16_t emtf_zones_;
            int16_t emtf_timezones_;

            bool flag_neighbor_;
            bool flag_substitute_;
            bool flag_valid_;
    };

    typedef std::vector<EMTFHit> EMTFHitCollection;

}  // namespace l1t:phase2

#endif  // DataFormats_L1TMuonPhase2_EMTFHit_h not defined
