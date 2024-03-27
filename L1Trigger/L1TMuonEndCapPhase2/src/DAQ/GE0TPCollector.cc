#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConfiguration.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/SubsystemTags.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/CSCUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/GE0TPCollector.h"

using namespace emtf::phase2;

GE0TPCollector::GE0TPCollector(const EMTFContext& context, edm::ConsumesCollector& i_consumes_collector)
    : context_(context),
      input_token_(i_consumes_collector.consumes<GE0Tag::collection_type>(context.config_.ge0_input_)) {}

void GE0TPCollector::collect(const edm::Event& i_event, BXTPCMap& bx_tpc_map) const {
  // Constants
  // First quarter of GE0 chamber (5 deg) and last quarter
  static const int me0_max_partition = 9;
  static const int me0_nstrips = 384;
  static const int me0_nphipositions = me0_nstrips * 2;
  static const int phiposition_q1 = me0_nphipositions / 4;
  static const int phiposition_q3 = (me0_nphipositions / 4) * 3;

  // Read GE0 digis
  TPCollection tpc;

  edm::Handle<GE0Tag::collection_type> me0_digis;
  i_event.getByToken(input_token_, me0_digis);

  auto chamber = me0_digis->begin();
  auto chend = me0_digis->end();

  for (; chamber != chend; ++chamber) {
    auto digi = (*chamber).second.first;
    auto dend = (*chamber).second.second;

    for (; digi != dend; ++digi) {
      tpc.emplace_back((*chamber).first, *digi);
    }
  }

  // Map to BX
  for (auto& tp_entry : tpc) {
    const auto& tp_det_id = tp_entry.tp_.detId<GEMDetId>();
    const ME0Data& tp_data = tp_entry.tp_.getME0Data();

    const int tp_region = tp_det_id.region();                    // 0: barrel, +/-1: endcap
    const int tp_endcap = (tp_region == -1) ? 2 : tp_region;     // 1: +endcap, 2: -endcap
    const int tp_endcap_pm = (tp_endcap == 2) ? -1 : tp_endcap;  // 1: +endcap, -1: -endcap
    const int tp_station = 1;                                    // ME0DetId station was always 1!
    const int tp_ring = 4;
    const int tp_layer = tp_det_id.layer();
    const int tp_roll = tp_det_id.roll();
    const int tp_me0_chamber = tp_det_id.chamber();

    const int tp_pad = tp_data.phiposition;
    const int tp_partition = tp_data.partition;
    const int tp_bx = tp_data.bx + this->context_.config_.me0_bx_shift_;

    // Reject if outside eta of 2.4
    if (tp_partition > me0_max_partition) {
      continue;
    }

    // Calculate EMTF Info
    // Split 20-deg chamber into 10-deg chamber
    // GE0 chamber is rotated by -5 deg relative to CSC chamber.
    // GE0 chamber 1 starts at -10 deg, CSC chamber 1 starts at -5 deg.
    const int tp_phiposition = tp_data.phiposition;  // in half-strip unit

    int tp_chamber = (tp_me0_chamber - 1) * 2 + 1;

    if (tp_endcap == 1) {
      // positive endcap
      // phiposition increases counter-clockwise
      if (tp_phiposition < phiposition_q1) {
        tp_chamber = csc::getNext10DegChamber(tp_chamber);
      } else if (tp_phiposition < phiposition_q3) {
        // Do nothing
      } else {
        tp_chamber = csc::getPrev10DegChamber(tp_chamber);
      }
    } else {
      // negative endcap
      // phiposition increases clockwise
      if (tp_phiposition < phiposition_q1) {
        tp_chamber = csc::getPrev10DegChamber(tp_chamber);
      } else if (tp_phiposition < phiposition_q3) {
        // Do nothing
      } else {
        tp_chamber = csc::getNext10DegChamber(tp_chamber);
      }
    }

    const int tp_sector = csc::getTriggerSector(tp_station, tp_ring, tp_chamber);
    const int tp_subsector = csc::getTriggerSubsector(tp_station, tp_chamber);
    const int tp_csc_id = csc::getId(tp_station, tp_ring, tp_chamber);
    const auto tp_csc_facing = csc::getFaceDirection(tp_station, tp_ring, tp_chamber);

    // Assertion checks
    emtf_assert(kMinEndcap <= tp_endcap && tp_endcap <= kMaxEndcap);
    emtf_assert(kMinTrigSector <= tp_sector && tp_sector <= kMaxTrigSector);
    emtf_assert((1 <= tp_subsector) and (tp_subsector <= 2));
    emtf_assert(tp_station == 1);
    emtf_assert(tp_ring == 4);
    emtf_assert((1 <= tp_chamber) and (tp_chamber <= 36));
    emtf_assert(1 <= tp_csc_id && tp_csc_id <= 3);
    emtf_assert(0 <= tp_pad && tp_pad <= 767);
    emtf_assert(0 <= tp_partition && tp_partition <= 15);

    // Add info
    tp_entry.info_.bx = tp_bx;

    tp_entry.info_.endcap = tp_endcap;
    tp_entry.info_.endcap_pm = tp_endcap_pm;
    tp_entry.info_.sector = tp_sector;
    tp_entry.info_.subsector = tp_subsector;
    tp_entry.info_.station = tp_station;
    tp_entry.info_.ring = tp_ring;
    tp_entry.info_.layer = tp_layer;
    tp_entry.info_.roll = tp_roll;
    tp_entry.info_.chamber = tp_chamber;

    tp_entry.info_.csc_id = tp_csc_id;
    tp_entry.info_.csc_facing = tp_csc_facing;

    bx_tpc_map[tp_bx].push_back(tp_entry);
  }
}
