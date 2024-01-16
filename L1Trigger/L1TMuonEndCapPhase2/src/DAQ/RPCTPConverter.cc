#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConfiguration.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/HostLut.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/SiteLut.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/ZoneLut.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/SubsystemTags.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/CSCUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/RPCUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/TPUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/RPCTPConverter.h"

using namespace emtf::phase2;

RPCTPConverter::RPCTPConverter(const EMTFContext& context, const int& endcap, const int& sector)
    : context_(context), endcap_(endcap), sector_(sector) {}

void RPCTPConverter::convert(const TriggerPrimitive& tp, const TPInfo& tp_info, EMTFHit& hit) const {
  // Unpack Id
  const auto& tp_hit_id = tp_info.hit_id;
  const auto& tp_segment_id = tp_info.segment_id;

  // Unpack trigger primitive
  const auto& tp_det_id = tp.detId<RPCDetId>();
  const auto& tp_data = tp.getRPCData();

  // Unpack detector info
  const auto tp_subsystem = L1TMuon::kRPC;

  const int tp_raw_id = tp_det_id.rawId();

  const int tp_endcap_pm = tp_info.endcap_pm;
  const int tp_subsector = tp_info.subsector;
  const int tp_station = tp_info.station;
  const int tp_ring = tp_info.ring;
  const int tp_roll = tp_info.roll;
  const int tp_layer = tp_info.layer;
  const int tp_chamber = tp_info.chamber;

  const auto tp_csc_facing = tp_info.csc_facing;
  const int tp_csc_id = tp_info.csc_id;
  const auto tp_rpc_type = tp_info.rpc_type;

  // Unpack data
  const int tp_strip = (tp_data.strip_low + tp_data.strip_hi) / 2;  // in full-strip unit
  const int tp_strip_lo = tp_data.strip_low;
  const int tp_strip_hi = tp_data.strip_hi;
  const int tp_clus_width = (tp_strip_hi - tp_strip_lo + 1);

  int tp_bend = 0;  // not applicable

  const int tp_bx = tp_info.bx;
  const float tp_time = tp_data.time;
  float tp_subbx_f32 = tp_time - (std::round(tp_time / 25.) * 25.);  // reduce range to [-12.5,12.5] ns
  int tp_subbx = static_cast<int>(std::round(tp_subbx_f32 * 16. / 25.));
  tp_subbx = std::clamp(tp_subbx, -8, 7);  // 4-bit, signed
  int tp_bx_check = static_cast<int>(std::round(tp_time / 25.));

  // Not sure why sometimes digi.time() returns 0?
  emtf_assert(((not(std::abs(tp_time) < 1e-6)) and (tp_bx == tp_bx_check)) or (std::abs(tp_time) < 1e-6));

  const auto tp_selection = tp_info.selection;

  // Use cluster width as quality.
  int tp_quality;

  if (tp_rpc_type == rpc::Type::kiRPC) {
    tp_quality = tp_clus_width;
  } else {
    tp_quality = tp_clus_width * 3 / 2;  // RPC strip pitch is 1.5 times the iRPC strip pitch.
  }

  // ID scheme used in FW
  const int tp_ilink = tp_info.ilink;

  // Get Global Coordinates
  float glob_phi;
  float glob_theta;
  double glob_rho;
  double glob_z;

  if (tp_rpc_type == rpc::Type::kiRPC) {
    // Handle iRPC Coordinates
    const RPCRoll* roll =
        dynamic_cast<const RPCRoll*>(this->context_.geometry_translator_.getRPCGeometry().roll(tp_det_id));
    const GlobalPoint& irpc_gp = roll->surface().toGlobal(LocalPoint(tp_data.x, tp_data.y, 0));

    glob_phi = tp::rad_to_deg(irpc_gp.phi().value());
    glob_theta = tp::rad_to_deg(irpc_gp.theta().value());
    glob_rho = irpc_gp.perp();
    glob_z = irpc_gp.z();
  } else {
    // Handle RPC Coordinates
    const GlobalPoint& gp = this->context_.geometry_translator_.getGlobalPoint(tp);
    glob_phi = tp::rad_to_deg(gp.phi().value());
    glob_theta = tp::rad_to_deg(gp.theta().value());
    glob_rho = gp.perp();
    glob_z = gp.z();
  }

  // Calculate EMTF Values
  const int emtf_phi = tp::calc_phi_int(sector_, glob_phi);
  const int emtf_bend = 0;
  const int emtf_theta = tp::calc_theta_int(tp_endcap_pm, glob_theta);
  const int emtf_qual = 0;
  const int emtf_site = context_.site_lut_.lookup({tp_subsystem, tp_station, tp_ring});
  const int emtf_host = context_.host_lut_.lookup({tp_subsystem, tp_station, tp_ring});
  const int emtf_zones = context_.zone_lut_.get_zones(emtf_host, emtf_theta);

  emtf_assert((0 <= emtf_phi) and (emtf_phi < 5040));
  emtf_assert((1 <= emtf_theta) and (emtf_theta < 128));

  // Get flags
  const bool tp_flag_neighbor = (tp_selection == TPSelection::kNeighbor);
  const bool tp_flag_substitute = tp_info.flag_substitute;
  const bool tp_flag_valid = tp_data.valid;

  // Set all the variables
  hit.setId(tp_hit_id);

  hit.setRawDetId(tp_raw_id);
  hit.setSubsystem(tp_subsystem);
  hit.setEndcap(tp_endcap_pm);
  hit.setSector(sector_);
  hit.setSubsector(tp_subsector);
  hit.setStation(tp_station);
  hit.setRing(tp_ring);
  hit.setLayer(tp_layer);
  hit.setChamber(tp_chamber);

  hit.setCscId(tp_csc_id);
  hit.setCscFR(tp_csc_facing == csc::Facing::kRear);

  hit.setStrip(tp_strip);
  hit.setStripLo(tp_strip_lo);
  hit.setStripHi(tp_strip_hi);

  hit.setWire1(tp_roll);
  hit.setWire2(0);

  hit.setBend(tp_bend);

  hit.setBx(tp_bx);
  hit.setSubbx(tp_subbx);

  hit.setQuality(tp_quality);
  hit.setPattern(0);

  hit.setGlobPhi(glob_phi);
  hit.setGlobTheta(glob_theta);
  hit.setGlobPerp(glob_rho);
  hit.setGlobZ(glob_z);
  hit.setGlobTime(tp_time);

  hit.setEmtfChamber(tp_ilink);
  hit.setEmtfSegment(tp_segment_id);
  hit.setEmtfPhi(emtf_phi);
  hit.setEmtfBend(emtf_bend);
  hit.setEmtfTheta1(emtf_theta);
  hit.setEmtfTheta2(0);
  hit.setEmtfQual1(emtf_qual);
  hit.setEmtfQual2(0);
  hit.setEmtfSite(emtf_site);
  hit.setEmtfHost(emtf_host);
  hit.setEmtfZones(emtf_zones);

  hit.setFlagNeighbor(tp_flag_neighbor);
  hit.setFlagSubstitute(tp_flag_substitute);
  hit.setFlagValid(tp_flag_valid);
}
