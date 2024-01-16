#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConfiguration.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/SubsystemTags.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/CSCUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/GE0TPSelector.h"

using namespace emtf::phase2;

GE0TPSelector::GE0TPSelector(const EMTFContext& context, const int& endcap, const int& sector)
    : context_(context), endcap_(endcap), sector_(sector) {}

void GE0TPSelector::select(const TriggerPrimitive& tp, TPInfo tp_info, ILinkTPCMap& ilink_tpc_map) const {
  emtf_assert(tp.subsystem() == L1TMuon::kME0);

  // Map GE0 trigger primitives to input links
  int ilink = get_input_link(tp, tp_info);  // Returns GE0 "link" index

  // Short-Circuit: Link not found (ilink = -1)
  if (ilink < 0) {
    return;
  }

  ilink_tpc_map[ilink].emplace_back(tp, tp_info);
}

// ===========================================================================
// Utils
// ===========================================================================
int GE0TPSelector::get_input_link(const TriggerPrimitive& tp, TPInfo& tp_info) const {
  int ilink = -1;

  // Unpack detector info
  const int tp_endcap = tp_info.endcap;
  const int tp_sector = tp_info.sector;
  const int tp_subsector = tp_info.subsector;
  const int tp_station = tp_info.station;
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
  ilink = calculate_input_link(tp_subsector, tp_csc_id, tp_selection);

  // Add selection info
  tp_info.ilink = ilink;
  tp_info.selection = tp_selection;

  return ilink;
}

int GE0TPSelector::calculate_input_link(const int& tp_subsector,
                                        const int& tp_csc_id,
                                        const TPSelection& tp_selection) const {
  int ilink = -1;

  // Links
  // GE0                  : 108..113
  // GE0 (N)              : 114

  if (tp_selection == TPSelection::kNative) {
    const int ilink_offset = 108;

    ilink = ilink_offset + ((tp_subsector - 1) * 3) + (tp_csc_id - 1);

    emtf_assert((108 <= ilink) && (ilink < 114));
  } else {
    const int ilink_offset = 114;

    ilink = ilink_offset;

    emtf_assert(ilink == ilink_offset);
  }

  return ilink;
}
