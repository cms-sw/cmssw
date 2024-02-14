#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConfiguration.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/SubsystemTags.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/CSCUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/CSCTPSelector.h"

using namespace emtf::phase2;

CSCTPSelector::CSCTPSelector(const EMTFContext& context, const int& endcap, const int& sector)
    : context_(context), endcap_(endcap), sector_(sector) {}

void CSCTPSelector::select(const TriggerPrimitive& tp, TPInfo tp_info, ILinkTPCMap& ilink_tpc_map) const {
  emtf_assert(tp.subsystem() == L1TMuon::kCSC);

  // Map CSC trigger primitives to input links
  int ilink = get_input_link(tp, tp_info);  // Returns CSC "link" index (0 - 53)

  // Short-Circuit: Link not found (ilink = -1)
  if (ilink < 0) {
    return;
  }

  // FIXME
  if (ilink_tpc_map[ilink].size() < 2) {
    ilink_tpc_map[ilink].emplace_back(tp, tp_info);
  } else {
    edm::LogWarning("L1TEMTFpp") << "\n******************* EMTF EMULATOR: SUPER-BIZZARE CASE *******************";
    edm::LogWarning("L1TEMTFpp") << "Found 3 CSC trigger primitives in the same chamber";

    for (int i_tp = 0; i_tp < 3; i_tp++) {
      const auto& tp_err = ((i_tp < 2) ? ilink_tpc_map[ilink].at(i_tp).tp_ : tp);

      edm::LogWarning("L1TEMTFpp") << "LCT #" << i_tp + 1 << ": BX " << tp_err.getBX() << ", endcap "
                                   << tp_err.detId<CSCDetId>().endcap() << ", sector "
                                   << tp_err.detId<CSCDetId>().triggerSector() << ", station "
                                   << tp_err.detId<CSCDetId>().station() << ", ring " << tp_err.detId<CSCDetId>().ring()
                                   << ", chamber " << tp_err.detId<CSCDetId>().chamber() << ", CSC ID "
                                   << tp_err.getCSCData().cscID << ": strip " << tp_err.getStrip() << ", wire "
                                   << tp_err.getWire();
    }

    edm::LogWarning("L1TEMTFpp") << "************************* ONLY KEEP FIRST TWO *************************\n\n";
  }
}

// ===========================================================================
// Utils
// ===========================================================================
int CSCTPSelector::get_input_link(const TriggerPrimitive& tp, TPInfo& tp_info) const {
  int ilink = -1;

  // Unpack detector info
  const int tp_endcap = tp_info.endcap;
  const int tp_sector = tp_info.sector;
  const int tp_subsector = tp_info.subsector;
  const int tp_station = tp_info.station;
  const int tp_ring = tp_info.ring;
  const int tp_csc_id = tp_info.csc_id;

  // Find selection type
  auto tp_selection = TPSelection::kNone;

  if (csc::is_in_sector(endcap_, sector_, tp_endcap, tp_sector)) {
    tp_selection = TPSelection::kNative;
  } else if (this->context_.config_.include_neighbor_en_ &&
             csc::is_in_neighbor_sector(endcap_, sector_, tp_endcap, tp_sector, tp_subsector, tp_station, tp_csc_id)) {
    tp_selection = TPSelection::kNeighbor;
  } else {  // Short-Circuit: tp_selection = TPSelection::kNone
    return ilink;
  }

  // Get chamber input link for this sector processor
  ilink = calculate_input_link(tp_subsector, tp_station, tp_ring, tp_csc_id, tp_selection);

  // Add selection info
  tp_info.ilink = ilink;
  tp_info.selection = tp_selection;

  return ilink;
}

// Returns CSC input "link".  Index used by FW for unique chamber identification.
int CSCTPSelector::calculate_input_link(const int& tp_subsector,
                                        const int& tp_station,
                                        const int& tp_ring,
                                        const int& tp_csc_id,
                                        const TPSelection& tp_selection) const {
  int ilink = -1;

  // Links
  // ME1,2,3,4        : 0..17,  18..26, 27..35, 36..44
  // ME1,2,3,4 (N)    : 45..47, 48..49, 50..51, 52..53

  if (tp_selection == TPSelection::kNative) {
    const int ilink_offset = 0;

    if (tp_station == 1) {
      ilink = ilink_offset + (tp_subsector - 1) * 9 + (tp_csc_id - 1);
    } else {
      ilink = ilink_offset + tp_station * 9 + (tp_csc_id - 1);
    }

    emtf_assert((0 <= ilink) && (ilink < 45));
  } else {
    const int ilink_offset = 45;

    if (tp_station == 1) {
      ilink = ilink_offset + ((tp_station - 1) * 2) + (tp_csc_id - 1) / 3;
    } else if (tp_ring == 1) {
      ilink = ilink_offset + ((tp_station - 1) * 2) + 1;
    } else {
      ilink = ilink_offset + ((tp_station - 1) * 2) + 2;
    }

    emtf_assert((45 <= ilink) && (ilink < 54));
  }

  return ilink;
}
