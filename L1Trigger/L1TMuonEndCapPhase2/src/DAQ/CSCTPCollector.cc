#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConfiguration.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/SubsystemTags.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/CSCTPCollector.h"

using namespace emtf::phase2;

CSCTPCollector::CSCTPCollector(const EMTFContext& context, edm::ConsumesCollector& i_consumes_collector)
    : context_(context),
      input_token_(i_consumes_collector.consumes<CSCTag::collection_type>(context.config_.csc_input_)) {}

void CSCTPCollector::collect(const edm::Event& i_event, BXTPCMap& bx_tpc_map) const {
  edm::Handle<CSCTag::collection_type> csc_digis;
  i_event.getByToken(input_token_, csc_digis);

  // Collect
  TPCollection tpc;

  auto chamber = csc_digis->begin();
  auto chend = csc_digis->end();

  for (; chamber != chend; ++chamber) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;

    for (; digi != dend; ++digi) {
      tpc.emplace_back((*chamber).first, *digi);
    }
  }

  // Find wires
  std::map<std::pair<uint32_t, uint16_t>, std::vector<uint16_t> > chamber_wires_map;

  for (auto& tp_entry : tpc) {
    const auto& tp_det_id = tp_entry.tp_.detId<CSCDetId>();

    const CSCData& tp_data = tp_entry.tp_.getCSCData();
    const int tp_bx = tp_data.bx + this->context_.config_.csc_bx_shift_;
    const int tp_wire = tp_data.keywire;

    auto key = std::make_pair(tp_det_id.rawId(), tp_bx);
    auto res = chamber_wires_map.find(key);

    if (res == chamber_wires_map.end()) {
      // Case: Chamber not found
      chamber_wires_map[key].push_back(tp_wire);
    } else {
      // Case: Chamber found
      // Lookup wire if found move on, otherwise add it.
      bool wire_found = false;

      auto& chamber_wires = res->second;

      for (const auto& a_wire : chamber_wires) {
        // Short-Circuit: If wire matches stop
        if (a_wire == tp_wire) {
          wire_found = true;
          break;
        }
      }

      // Case: Wire not found, add it.
      if (!wire_found) {
        chamber_wires.push_back(tp_wire);
      }
    }
  }

  // Map to BX
  for (auto& tp_entry : tpc) {
    const auto& tp_det_id = tp_entry.tp_.detId<CSCDetId>();
    const CSCData& tp_data = tp_entry.tp_.getCSCData();

    const int tp_endcap = tp_det_id.endcap();                    // 1: +endcap, 2: -endcap
    const int tp_endcap_pm = (tp_endcap == 2) ? -1 : tp_endcap;  // 1: +endcap, -1: -endcap
    const int tp_sector = tp_det_id.triggerSector();
    const int tp_station = tp_det_id.station();
    const int tp_ring = tp_det_id.ring();
    const int tp_chamber = tp_det_id.chamber();
    const int tp_layer = tp_det_id.layer();

    const int tp_csc_id = tp_data.cscID;

    const int tp_bx = tp_data.bx + this->context_.config_.csc_bx_shift_;

    // Get wires
    int tp_wire1 = tp_data.keywire;
    int tp_wire2 = -1;

    auto tp_wire_key = std::make_pair(tp_det_id.rawId(), tp_bx);
    const auto& tp_wires = chamber_wires_map.at(tp_wire_key);

    emtf_assert((!tp_wires.empty()) && (tp_wires.size() <= 2));

    if (tp_wires.size() > 1) {
      tp_wire1 = tp_wires.at(0);
      tp_wire2 = tp_wires.at(1);
    }

    // Calculate detector info
    const int tp_subsector = csc::get_trigger_subsector(tp_station, tp_chamber);
    const auto tp_face_dir = csc::get_face_direction(tp_station, tp_ring, tp_chamber);

    // Assertion checks
    const auto& [max_strip, max_wire] = csc::get_max_strip_and_wire(tp_station, tp_ring);
    const auto& [max_pattern, max_quality] = csc::get_max_pattern_and_quality(tp_station, tp_ring);

    emtf_assert(kMinEndcap <= tp_endcap && tp_endcap <= kMaxEndcap);
    emtf_assert(kMinTrigSector <= tp_sector && tp_sector <= kMaxTrigSector);
    emtf_assert((0 <= tp_subsector) and (tp_subsector <= 2));
    emtf_assert(1 <= tp_station && tp_station <= 4);
    emtf_assert(1 <= tp_ring && tp_ring <= 4);
    emtf_assert(1 <= tp_chamber && tp_chamber <= 36);
    emtf_assert(1 <= tp_csc_id && tp_csc_id <= 9);
    emtf_assert(tp_data.strip < max_strip);
    emtf_assert(tp_data.keywire < max_wire);
    emtf_assert(tp_data.pattern < max_pattern);
    emtf_assert(0 < tp_data.quality && tp_data.quality < max_quality);
    emtf_assert(tp_data.valid);

    // Check for corrupted LCT data. Data corruption could occur due to software
    // or hardware issues, if corrupted, reject the LCT.
    if (!(tp_data.strip < max_strip)) {
      edm::LogWarning("L1TEMTFpp") << "Found error in LCT strip: " << tp_data.strip << " (allowed range: 0-"
                                   << max_strip - 1 << ").";

      edm::LogWarning("L1TEMTFpp")
          << "From endcap " << tp_endcap << ", sector " << tp_sector << ", station " << tp_station << ", ring "
          << tp_ring << ", cscid " << tp_csc_id
          << ". (Note that this LCT may be reported multiple times. See source code for explanations.)";

      continue;
    }

    if (!(tp_data.keywire < max_wire)) {
      edm::LogWarning("L1TEMTFpp") << "Found error in LCT wire: " << tp_data.keywire << " (allowed range: 0-"
                                   << max_wire - 1 << ").";

      edm::LogWarning("L1TEMTFpp")
          << "From endcap " << tp_endcap << ", sector " << tp_sector << ", station " << tp_station << ", ring "
          << tp_ring << ", cscid " << tp_csc_id
          << ". (Note that this LCT may be reported multiple times. See source code for explanations.)";

      continue;
    }

    if (!(tp_data.valid == true)) {
      edm::LogWarning("L1TEMTFpp") << "Found error in LCT valid: " << tp_data.valid << " (allowed value: 1).";

      edm::LogWarning("L1TEMTFpp")
          << "From endcap " << tp_endcap << ", sector " << tp_sector << ", station " << tp_station << ", ring "
          << tp_ring << ", cscid " << tp_csc_id
          << ". (Note that this LCT may be reported multiple times. See source code for explanations.)";

      continue;
    }

    if (!(tp_data.pattern < max_pattern)) {
      edm::LogWarning("L1TEMTFpp") << "Found error in LCT pattern: " << tp_data.pattern << " (allowed range: 0-"
                                   << max_pattern - 1 << ").";

      edm::LogWarning("L1TEMTFpp")
          << "From endcap " << tp_endcap << ", sector " << tp_sector << ", station " << tp_station << ", ring "
          << tp_ring << ", cscid " << tp_csc_id
          << ". (Note that this LCT may be reported multiple times. See source code for explanations.)";

      continue;
    }

    if (!(0 < tp_data.quality && tp_data.quality < max_quality)) {
      edm::LogWarning("L1TEMTFpp") << "Found error in LCT quality: " << tp_data.quality << " (allowed range: 1-"
                                   << max_quality - 1 << ").";

      edm::LogWarning("L1TEMTFpp")
          << "From endcap " << tp_endcap << ", sector " << tp_sector << ", station " << tp_station << ", ring "
          << tp_ring << ", cscid " << tp_csc_id
          << ". (Note that this LCT may be reported multiple times. See source code for explanations.)";

      continue;
    }

    // Add info
    tp_entry.info_.bx = tp_bx;

    tp_entry.info_.endcap = tp_endcap;
    tp_entry.info_.endcap_pm = tp_endcap_pm;
    tp_entry.info_.sector = tp_sector;
    tp_entry.info_.subsector = tp_subsector;
    tp_entry.info_.station = tp_station;
    tp_entry.info_.ring = tp_ring;
    tp_entry.info_.chamber = tp_chamber;
    tp_entry.info_.layer = tp_layer;

    tp_entry.info_.csc_id = tp_csc_id;
    tp_entry.info_.csc_facing = tp_face_dir;
    tp_entry.info_.csc_first_wire = tp_wire1;
    tp_entry.info_.csc_second_wire = tp_wire2;

    bx_tpc_map[tp_bx].push_back(tp_entry);
  }
}
