#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConfiguration.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/GEMTPCollector.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/SubsystemTags.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"

using namespace emtf::phase2;

GEMTPCollector::GEMTPCollector(
        const EMTFContext& context,
        edm::ConsumesCollector& i_consumes_collector):
    context_(context),
    input_token_(i_consumes_collector.consumes<GEMTag::collection_type>(
                context.pset_.getParameter<edm::InputTag>("GEMInput")))
{
    // Do Nothing
}

GEMTPCollector::~GEMTPCollector() {
    // Do Nothing
}

void GEMTPCollector::collect(
        const edm::Event& i_event,
        BXTPCMap& bx_tpc_map
) const {
    // Constants
    static const int max_delta_roll = 1;
    static const int max_delta_pad_ge11 = 4;
    static const int max_delta_pad_ge21 = 4;

    // Read GEM digis
    TPCollection tpc;

    edm::Handle<GEMTag::collection_type> gem_digis;
    i_event.getByToken(input_token_, gem_digis);

    auto chamber = gem_digis->begin();
    auto chend = gem_digis->end();

    for (; chamber != chend; ++chamber) {
        auto digi = (*chamber).second.first;
        auto dend = (*chamber).second.second;

        for (; digi != dend; ++digi) {
            // Short-Circuit: Ignore invalid digis
            bool tp_valid = (*digi).isValid();

            if (!tp_valid) {
                continue;
            }

            // Append digi
            tpc.emplace_back((*chamber).first, *digi);
        }
    }

    // Find Copads
    std::map<
        std::pair<uint32_t, uint16_t>, 
        std::vector<std::array<uint16_t, 3>>
    > chamber_copads_map;

    for (auto& tp_entry : tpc) {
        const auto& tp_det_id = tp_entry.tp_.detId<GEMDetId>();
        const GEMData& tp_data = tp_entry.tp_.getGEMData();

        const int tp_region = tp_det_id.region();                   // 0: barrel, +/-1: endcap
        const int tp_station = tp_det_id.station();
        const int tp_ring = tp_det_id.ring();
        const int tp_chamber = tp_det_id.chamber();
        const int tp_layer = tp_det_id.layer();

        const uint16_t tp_roll = tp_det_id.roll();
        const uint16_t tp_pad_lo = tp_data.pad_low;
        const uint16_t tp_pad_hi = tp_data.pad_hi;

        const int tp_bx = tp_data.bx + CONFIG.gem_bx_shift_;

        GEMDetId tp_mod_det_id(tp_region, tp_ring, tp_station, 0, tp_chamber, 0); 
        auto key = std::make_pair(tp_mod_det_id.rawId(), tp_bx);

        if (tp_layer == 1) {            
            // Layer 1 is incidence
            // If key does not exist, insert an empty vector. If key exists, do nothing.
            decltype(chamber_copads_map)::mapped_type copads;
            chamber_copads_map.insert({key, copads});
        } else if (tp_layer == 2) {
            // Layer 2 is coincidence
            decltype(chamber_copads_map)::mapped_type::value_type copad{{
                tp_roll, 
                tp_pad_lo, 
                tp_pad_hi
            }};
            chamber_copads_map[key].push_back(copad);
        }
    }

    // Map to BX
    for (auto& tp_entry : tpc) {
        const auto& tp_det_id = tp_entry.tp_.detId<GEMDetId>();
        const GEMData& tp_data = tp_entry.tp_.getGEMData();

        const int tp_region = tp_det_id.region();                   // 0: barrel, +/-1: endcap
        const int tp_endcap = (tp_region == -1) ? 2 : tp_region;    // 1: +endcap, 2: -endcap
        const int tp_endcap_pm = (tp_endcap == 2) ? -1 : tp_endcap; // 1: +endcap, -1: -endcap
        const int tp_station = tp_det_id.station();
        const int tp_ring = tp_det_id.ring();
        const int tp_layer = tp_det_id.layer();
        const int tp_chamber = tp_det_id.chamber();

        const int tp_roll = tp_det_id.roll();
        const int tp_pad = (tp_data.pad_low + tp_data.pad_hi) / 2;
        const int tp_pad_lo = tp_data.pad_low;
        const int tp_pad_hi = tp_data.pad_hi;

        const int tp_bx = tp_data.bx + CONFIG.gem_bx_shift_;

        // Get Copad Info
        GEMDetId tp_mod_det_id(tp_region, tp_ring, tp_station, 0, tp_chamber, 0); 
        auto tp_copads_key = std::make_pair(tp_mod_det_id.rawId(), tp_bx);
        auto tp_copads = chamber_copads_map.at(tp_copads_key);

        // Check Copads
        bool tp_is_substitute = false;

        if (tp_layer == 1) {
            // layer 1 is used as incidence
            const bool is_ge21 = (tp_station == 2);
            
            auto match_fn = [&tp_roll, &tp_pad_lo, &tp_pad_hi, &is_ge21](const std::array<uint16_t, 3>& elem) {
                // Unpack entry
                // Compare roll and (pad_lo, pad_hi)-range with tolerance
                const auto& [c_roll_tmp, c_pad_lo_tmp, c_pad_hi_tmp] = elem;
                int c_roll_lo = static_cast<int>(c_roll_tmp) - max_delta_roll;
                int c_roll_hi = static_cast<int>(c_roll_tmp) + max_delta_roll;
                int c_pad_lo = static_cast<int>(c_pad_lo_tmp) - (is_ge21 ? max_delta_pad_ge21 : max_delta_pad_ge11);
                int c_pad_hi = static_cast<int>(c_pad_hi_tmp) + (is_ge21 ? max_delta_pad_ge21 : max_delta_pad_ge11);

                // Two ranges overlap if (range_a_lo <= range_b_hi) and (range_a_hi >= range_b_lo)
                return (tp_roll <= c_roll_hi) and (tp_roll >= c_roll_lo) and (tp_pad_lo <= c_pad_hi) and (tp_pad_hi >= c_pad_lo);
            };

            auto match = std::find_if(tp_copads.begin(), tp_copads.end(), match_fn);

            if (match != tp_copads.end()) {
                // Has copad
                tp_is_substitute = false;
            } else if (tp_copads.empty()) {
                // Kinda has copad
                tp_is_substitute = true;
            } else {
                // Short-Circuit: Didn't find coincidence
                continue;
            }
        } else if (tp_layer == 2) {
            // Short-Circuit: layer 2 is used as coincidence
            continue;
        }

        // Calculate EMTF Info
        const int tp_sector = csc::get_trigger_sector(tp_station, tp_ring, tp_chamber);
        const int tp_subsector = csc::get_trigger_subsector(tp_station, tp_chamber);
        const int tp_csc_id = csc::get_id(tp_station, tp_ring, tp_chamber);
        const auto tp_csc_facing = csc::get_face_direction(tp_station, tp_ring, tp_chamber);

        // Assertion checks
        emtf_assert(kMinEndcap <= tp_endcap && tp_endcap <= kMaxEndcap);
        emtf_assert(kMinTrigSector <= tp_sector && tp_sector <= kMaxTrigSector);
        emtf_assert((0 <= tp_subsector) and (tp_subsector <= 2));
        emtf_assert(1 <= tp_station && tp_station <= 2);
        emtf_assert(tp_ring == 1);
        emtf_assert((1 <= tp_chamber) and (tp_chamber <= 36));
        emtf_assert(1 <= tp_csc_id && tp_csc_id <= 3);
        emtf_assert(tp_station == 1 or (1 <= tp_roll && tp_roll <= 16));
        emtf_assert(tp_station != 1 or (1 <= tp_roll && tp_roll <= 8));
        emtf_assert(1 <= tp_layer && tp_layer <= 2);
        emtf_assert((tp_station == 1 && 0 <= tp_pad && tp_pad <= 191) || (tp_station != 1));
        emtf_assert((tp_station == 2 && 0 <= tp_pad && tp_pad <= 383) || (tp_station != 2));

        // Add info
        tp_entry.info_.bx = tp_bx;

        tp_entry.info_.endcap = tp_endcap;
        tp_entry.info_.endcap_pm = tp_endcap_pm;
        tp_entry.info_.sector = tp_sector;
        tp_entry.info_.subsector = tp_subsector;
        tp_entry.info_.station = tp_station;
        tp_entry.info_.ring = tp_ring;
        tp_entry.info_.roll = tp_roll;
        tp_entry.info_.layer = tp_layer;
        tp_entry.info_.chamber = tp_chamber;

        tp_entry.info_.csc_id = tp_csc_id;
        tp_entry.info_.csc_facing = tp_csc_facing;

        tp_entry.info_.flag_substitute = tp_is_substitute;

        bx_tpc_map[tp_bx].push_back(tp_entry);
    }
}

