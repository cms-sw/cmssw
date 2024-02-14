#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConfiguration.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/RPCTPCollector.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/SubsystemTags.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/RPCUtils.h"

using namespace emtf::phase2;

RPCTPCollector::RPCTPCollector(const EMTFContext& context, edm::ConsumesCollector& i_consumes_collector)
    : context_(context),
      input_token_(i_consumes_collector.consumes<RPCTag::rechit_collection_type>(context.config_.rpc_input_)) {}

void RPCTPCollector::collect(const edm::Event& i_event, BXTPCMap& bx_tpc_map) const {
  // Constants
  static const int clus_width_cut = 4;
  static const int clus_width_cut_irpc = 6;

  // Read RPC digis
  TPCollection tpc;

  edm::Handle<RPCTag::rechit_collection_type> rpc_digis;
  i_event.getByToken(input_token_, rpc_digis);

  auto digi = rpc_digis->begin();
  auto digi_end = rpc_digis->end();

  for (; digi != digi_end; ++digi) {
    tpc.emplace_back(digi->rpcId(), *digi);
  }

  // Map to BX
  for (auto& tp_entry : tpc) {
    const auto& tp_det_id = tp_entry.tp_.detId<RPCDetId>();
    const RPCData& tp_data = tp_entry.tp_.getRPCData();

    const int tp_region = tp_det_id.region();                    // 0: barrel, +/-1: endcap
    const int tp_endcap = (tp_region == -1) ? 2 : tp_region;     // 1: +endcap, 2: -endcap
    const int tp_endcap_pm = (tp_endcap == 2) ? -1 : tp_endcap;  // 1: +endcap, -1: -endcap

    // RPC sector is rotated by -20 deg relative to CSC sector.
    // RPC sector 1 starts at -5 deg, CSC sector 1 starts at 15 deg.
    const int tp_rpc_sector = tp_det_id.sector();  // 1 - 6 (60 degrees in phi, sector 1 begins at -5 deg)

    // RPC subsector is defined differently than CSC subsector.
    // RPC subsector is used to label the chamber within a sector.
    const int tp_rpc_subsector = tp_det_id.subsector();

    const int tp_station = tp_det_id.station();  // 1 - 4
    const int tp_ring = tp_det_id.ring();        // 2 - 3 (increasing theta)
    const int tp_roll =
        tp_det_id.roll();  // 1 - 3 (decreasing theta; aka A - C; space between rolls is 9 - 15 in theta_fp)
    const int tp_layer = tp_det_id.layer();  // Always 1 in the Endcap, 1 or 2 in the Barrel

    const int tp_strip = (tp_data.strip_low + tp_data.strip_hi) / 2;  // in full-strip unit
    const int tp_strip_lo = tp_data.strip_low;
    const int tp_strip_hi = tp_data.strip_hi;
    const int tp_clus_width = (tp_strip_hi - tp_strip_lo + 1);

    const bool tp_is_CPPF = tp_data.isCPPF;

    const int tp_bx = tp_data.bx + this->context_.config_.rpc_bx_shift_;

    // Check Ring
    bool tp_is_substitute = (tp_ring == 3);

    // Calculate type
    const bool tp_is_barrel = (tp_region == 0);

    rpc::Type tp_rpc_type;

    if ((!tp_is_barrel) && (tp_station >= 3) && (tp_ring == 1)) {
      tp_rpc_type = rpc::Type::kiRPC;
    } else {
      tp_rpc_type = rpc::Type::kRPC;
    }

    // Short-Circuit: Skip Barrel RPC (region = 0)
    if (tp_region == 0) {
      continue;
    }

    // Short-Circuit: Skip Overlap region (RE1/3, RE2/3)
    if (tp_station <= 2 && tp_ring == 3) {
      continue;
    }

    // Short-Circuit: Reject wide clusters
    if (tp_rpc_type == rpc::Type::kiRPC) {
      if (tp_clus_width > clus_width_cut_irpc) {
        continue;
      }
    } else {
      if (tp_clus_width > clus_width_cut) {
        continue;
      }
    }

    // Calculate EMTF Info
    int tp_chamber;

    if (tp_rpc_type == rpc::Type::kiRPC) {
      tp_chamber = (tp_rpc_sector - 1) * 3 + tp_rpc_subsector;
    } else {
      tp_chamber = (tp_rpc_sector - 1) * 6 + tp_rpc_subsector;
    }

    const int tp_sector = csc::get_trigger_sector(tp_station, tp_ring, tp_chamber);
    const int tp_subsector = csc::get_trigger_subsector(tp_station, tp_chamber);
    const int tp_csc_id = csc::get_id(tp_station, tp_ring, tp_chamber);
    const auto tp_csc_facing = csc::get_face_direction(tp_station, tp_ring, tp_chamber);

    // Assertion checks
    emtf_assert(kMinEndcap <= tp_endcap && tp_endcap <= kMaxEndcap);
    emtf_assert(kMinTrigSector <= tp_sector && tp_sector <= kMaxTrigSector);
    emtf_assert(0 <= tp_subsector && tp_subsector <= 2);
    emtf_assert(1 <= tp_station && tp_station <= 4);
    emtf_assert(1 <= tp_chamber && tp_chamber <= 36);
    emtf_assert((1 <= tp_csc_id) and (tp_csc_id <= 9));

    if (tp_rpc_type == rpc::Type::kiRPC) {
      emtf_assert(tp_ring == 1);
      emtf_assert(1 <= tp_roll && tp_roll <= 5);
      emtf_assert(1 <= tp_strip && tp_strip <= 96);
    } else {
      emtf_assert(2 <= tp_ring && tp_ring <= 3);
      emtf_assert(1 <= tp_roll && tp_roll <= 3);
      emtf_assert(tp_is_CPPF || (1 <= tp_strip && tp_strip <= 32));
    }

    emtf_assert(tp_data.valid);

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

    tp_entry.info_.rpc_type = tp_rpc_type;

    tp_entry.info_.flag_substitute = tp_is_substitute;

    bx_tpc_map[tp_bx].push_back(tp_entry);
  }
}
