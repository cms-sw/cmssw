#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConfiguration.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/SubsystemTags.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/CSCUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/GEMTPSelector.h"

using namespace emtf::phase2;

GEMTPSelector::GEMTPSelector(const EMTFContext& context, const int& endcap, const int& sector)
    : context_(context), endcap_(endcap), sector_(sector) {}

void GEMTPSelector::select(const TriggerPrimitive& tp, TPInfo tp_info, ILinkTPCMap& ilink_tpc_map) const {
  emtf_assert(tp.subsystem() == L1TMuon::kGEM);

  // Map GEM trigger primitives to input links
  int ilink = getInputLink(tp, tp_info);  // Returns GEM "link" index

  // Short-Circuit: Link not found (ilink = -1)
  if (ilink < 0) {
    return;
  }

  ilink_tpc_map[ilink].emplace_back(tp, tp_info);
}

// ===========================================================================
// Utils
// ===========================================================================
int GEMTPSelector::getInputLink(const TriggerPrimitive& tp, TPInfo& tp_info) const {
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

  if (csc::isTPInSector(endcap_, sector_, tp_endcap, tp_sector)) {
    tp_selection = TPSelection::kNative;
  } else if (this->context_.config_.include_neighbor_en_ &&
             csc::isTPInNeighborSector(endcap_, sector_, tp_endcap, tp_sector, tp_subsector, tp_station, tp_csc_id)) {
    tp_selection = TPSelection::kNeighbor;
  } else {  // Short-Circuit: tp_selection = TPSelection::kNone
    return ilink;
  }

  // Get chamber input link for this sector processor
  ilink = calcInputLink(tp_subsector, tp_station, tp_ring, tp_csc_id, tp_selection);

  // Add selection info
  tp_info.ilink = ilink;
  tp_info.selection = tp_selection;

  return ilink;
}

int GEMTPSelector::calcInputLink(const int& tp_subsector,
                                 const int& tp_station,
                                 const int& tp_ring,
                                 const int& tp_csc_id,
                                 const TPSelection& tp_selection) const {
  int ilink = -1;

  // Links
  // RE1,2,3,4 + GE1,2        : 54..71, 72..80, 81..89, 90..98
  // RE1,2,3,4 + GE1,2 (N)    : 99..101, 102..103, 104..105, 106..107

  if (tp_selection == TPSelection::kNative) {
    const int ilink_offset = 54;

    if (tp_station == 1) {
      ilink = ilink_offset + (tp_subsector - 1) * 9 + (tp_csc_id - 1);
    } else {
      ilink = ilink_offset + tp_station * 9 + (tp_csc_id - 1);
    }

    emtf_assert((54 <= ilink) && (ilink < 99));
  } else {
    const int ilink_offset = 99;

    if (tp_station == 1) {
      ilink = ilink_offset + ((tp_station - 1) * 2) + ((tp_csc_id - 1) / 3);
    } else if (tp_ring == 1) {
      ilink = ilink_offset + ((tp_station - 1) * 2) + 1;
    } else {
      ilink = ilink_offset + ((tp_station - 1) * 2) + 2;
    }

    emtf_assert((99 <= ilink) && (ilink < 108));
  }

  return ilink;
}
