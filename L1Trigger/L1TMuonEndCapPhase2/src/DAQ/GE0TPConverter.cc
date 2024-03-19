#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConfiguration.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/HostLut.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/SiteLut.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Data/ZoneLut.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/SubsystemTags.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/CSCUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/TPUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/GE0TPConverter.h"

using namespace emtf::phase2;

GE0TPConverter::GE0TPConverter(const EMTFContext& context, const int& endcap, const int& sector)
    : context_(context), endcap_(endcap), sector_(sector) {}

void GE0TPConverter::convert(const TriggerPrimitive& tp, const TPInfo& tp_info, EMTFHit& hit) const {
  // Unpack Id
  const auto& tp_hit_id = tp_info.hit_id;
  const int tp_segment_id = tp_info.segment_id;

  // Unpack trigger primitive
  const auto& tp_det_id = tp.detId<GEMDetId>();
  const auto& tp_data = tp.getME0Data();

  // Unpack detector info
  const auto tp_subsystem = L1TMuon::kME0;

  const int tp_raw_id = tp_det_id.rawId();

  const int tp_endcap_pm = tp_info.endcap_pm;
  const int tp_subsector = tp_info.subsector;
  const int tp_station = tp_info.station;
  const int tp_ring = tp_info.ring;
  const int tp_layer = tp_info.layer;
  // const int tp_chamber = tp_info.chamber;

  const int tp_csc_id = tp_info.csc_id;
  const auto tp_csc_facing = tp_info.csc_facing;

  // Unpack data
  const int tp_phiposition = tp_data.phiposition;  // in half-strip unit
  const int tp_partition = tp_data.partition;      // in half-roll unit

  const int tp_bend = static_cast<int>(tp_data.deltaphi) * (tp_data.bend == 0 ? 1 : -1);

  const int tp_bx = tp_info.bx;
  const int tp_subbx = 0;   // no fine resolution timing
  const float tp_time = 0;  // no fine resolution timing.

  const auto tp_selection = tp_info.selection;

  const int tp_quality = tp_data.quality;

  // ID scheme used in FW
  const int tp_ilink = tp_info.ilink;

  // Get Global Coordinates
  const GlobalPoint& gp = this->context_.geometry_translator_.getGlobalPoint(tp);
  const float glob_phi = tp::radToDeg(gp.phi().value());
  const float glob_theta = tp::radToDeg(gp.theta().value());
  const double glob_rho = gp.perp();
  const double glob_z = gp.z();

  // Calculate EMTF Values
  const int emtf_phi = tp::calcPhiInt(sector_, glob_phi);
  const int emtf_bend = std::clamp(tp_bend / 2, -64, 63);  // 7-bit, signed
  const int emtf_theta = tp::calcThetaInt(tp_endcap_pm, glob_theta);
  const int emtf_qual = std::clamp(tp_quality, 0, 15);  // 4-bit, unsigned
  const int emtf_site = context_.site_lut_.lookup({tp_subsystem, tp_station, tp_ring});
  const int emtf_host = context_.host_lut_.lookup({tp_subsystem, tp_station, tp_ring});
  const int emtf_zones = context_.zone_lut_.getZones(emtf_host, emtf_theta);

  emtf_assert((0 <= emtf_phi) and (emtf_phi < 5040));
  emtf_assert((1 <= emtf_theta) and (emtf_theta < 128));

  // Get flags
  const bool tp_flag_neighbor = (tp_selection == TPSelection::kNeighbor);
  const bool tp_flag_substitute = tp_info.flag_substitute;
  const bool tp_flag_valid = true;  // given by digi, not trigger_primitive :(

  // Set properties
  hit.setId(tp_hit_id);

  hit.setRawDetId(tp_raw_id);
  hit.setSubsystem(L1TMuon::kME0);
  hit.setEndcap(tp_endcap_pm);
  hit.setSector(sector_);
  hit.setSubsector(tp_subsector);
  hit.setStation(tp_station);
  hit.setRing(tp_ring);
  hit.setLayer(tp_layer);
  hit.setChamber(tp_det_id.chamber());  // Save original chamber

  hit.setCscId(tp_csc_id);
  hit.setCscFR(tp_csc_facing == csc::Facing::kRear);

  hit.setStrip(tp_phiposition);
  hit.setStripLo(tp_phiposition);
  hit.setStripHi(tp_phiposition);

  hit.setWire1(tp_partition);
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
