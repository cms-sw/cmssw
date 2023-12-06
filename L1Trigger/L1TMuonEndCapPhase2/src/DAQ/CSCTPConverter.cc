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

#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/CSCTPConverter.h"

using namespace emtf::phase2;

CSCTPConverter::CSCTPConverter(const EMTFContext& context,
        const int& endcap, const int& sector):
    context_(context),
    endcap_(endcap),
    sector_(sector)
{
    // Do Nothing
}

CSCTPConverter::~CSCTPConverter() {
    // Do Nothing
}

void CSCTPConverter::convert(
        const TriggerPrimitive& tp,
        const TPInfo& tp_info,
        EMTFHit& hit
) const {
    // Unpack Id
    const auto& tp_hit_id = tp_info.hit_id;
    const auto& tp_segment_id = tp_info.segment_id;

    // Unpack trigger primitive
    const auto& tp_det_id = tp.detId<CSCDetId>();
    const auto& tp_data = tp.getCSCData();

    // Unpack detector info
    const auto tp_subsystem = L1TMuon::kCSC;

    const int tp_raw_id = tp_det_id.rawId();

    const int tp_endcap_pm = tp_info.endcap_pm;
    const int tp_subsector = tp_info.subsector;
    const int tp_chamber = tp_info.chamber;
    const int tp_station = tp_info.station;
    const int tp_ring = tp_info.ring;
    const int tp_layer = tp_info.layer;

    const int tp_csc_id = tp_info.csc_id;
    const auto tp_csc_facing = tp_info.csc_facing;

    // Unpack data
    const int tp_strip = tp_data.strip;
    const int tp_strip_quart_bit = tp_data.strip_quart_bit;
    const int tp_strip_eighth_bit = tp_data.strip_eighth_bit;
    const int tp_strip_quart = tp_data.strip_quart;
    const int tp_strip_eighth = tp_data.strip_eighth;

    const int tp_wire1 = tp_info.csc_first_wire;
    const int tp_wire2 = tp_info.csc_second_wire;

    int tp_bend;
    const int tp_slope = tp_data.slope;

    const int tp_bx = tp_info.bx;
    const int tp_subbx = 0;  // no fine resolution timing
    const float tp_time = 0;  // no fine resolution timing. Note: Check with Efe, Jia Fu gets this from digi directly.

    const auto tp_selection = tp_info.selection;

    const int tp_pattern = tp_data.pattern;
    const int tp_quality = 6;

    // Apply CSC Run 2 pattern -> bend conversion
    // Override tp_bend
    constexpr int tp_bend_lut_size = 11;
    constexpr int tp_bend_lut[tp_bend_lut_size] = {-5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0};
    emtf_assert(tp_pattern < tp_bend_lut_size);
    tp_bend = tp_bend_lut[tp_pattern];
    tp_bend *= tp_endcap_pm;  // sign flip depending on endcap

    // ID scheme used in FW
    const int tp_ilink = tp_info.ilink;

    // Get Global Coordinates
    const GlobalPoint& gp_w1 = GEOM.getGlobalPoint(tp);
    const float glob_phi_w1 = tp::rad_to_deg(gp_w1.phi().value());
    const float glob_theta_w1 = tp::rad_to_deg(gp_w1.theta().value());
    const double glob_rho_w1 = gp_w1.perp();
    const double glob_z_w1 = gp_w1.z();

    // Calculate EMTF Values
    const int emtf_phi_w1 = tp::calc_phi_int(sector_, glob_phi_w1);
    const int emtf_bend_w1 = std::clamp(tp_bend * 4, -16, 15);  // 5-bit, signed
    const int emtf_theta_w1 = tp::calc_theta_int(tp_endcap_pm, glob_theta_w1);
    const int emtf_qual_w1 = std::clamp(tp_quality, 0, 15);  // 4-bit, unsigned
    const int emtf_site_w1 = context_.site_lut_.lookup({tp_subsystem, tp_station, tp_ring});
    const int emtf_host_w1 = context_.host_lut_.lookup({tp_subsystem, tp_station, tp_ring});
    const int emtf_zones_w1 = context_.zone_lut_.get_zones(emtf_host_w1, emtf_theta_w1);

    // Calculated Ambiguous Info
    int emtf_theta_w2 = 0;
    int emtf_qual_w2 = tp_pattern;

    if (tp_wire2 > -1) {
        auto tp_w2 = tp;

        tp_w2.accessCSCData().keywire = tp_wire2;

        const GlobalPoint& gp_w2 = GEOM.getGlobalPoint(tp_w2);
        const double glob_theta_w2 = tp::rad_to_deg(gp_w2.theta().value());

        emtf_theta_w2 = tp::calc_theta_int(tp_endcap_pm, glob_theta_w2);
    }

    emtf_assert((0 <= emtf_phi_w1) and (emtf_phi_w1 < 5040));
    emtf_assert((1 <= emtf_theta_w1) and (emtf_theta_w1 < 128));
    emtf_assert((0 <= emtf_theta_w2) and (emtf_theta_w2 < 128));

    // Get flags
    const bool tp_flag_neighbor = (tp_selection == TPSelection::kNeighbor);
    const bool tp_flag_substitute = tp_info.flag_substitute;
    const bool tp_flag_valid = tp_data.valid;

    // Set properties
    hit.setId(tp_hit_id);

    hit.setRawDetId(tp_raw_id);
    hit.setSubsystem(L1TMuon::kCSC);
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
    hit.setStripLo(tp_strip);
    hit.setStripHi(tp_strip);
    hit.setStripQuart(tp_strip_quart);
    hit.setStripEighth(tp_strip_eighth);
    hit.setStripQuartBit(tp_strip_quart_bit);
    hit.setStripEighthBit(tp_strip_eighth_bit);

    hit.setWire1(tp_wire1);
    hit.setWire2(tp_wire2);

    hit.setBend(tp_bend);
    hit.setSlope(tp_slope);

    hit.setBx(tp_bx);
    hit.setSubbx(tp_subbx);

    hit.setQuality(tp_quality);
    hit.setPattern(tp_pattern);

    hit.setGlobPhi(glob_phi_w1);
    hit.setGlobTheta(glob_theta_w1);
    hit.setGlobPerp(glob_rho_w1);
    hit.setGlobZ(glob_z_w1);
    hit.setGlobTime(tp_time);

    hit.setEmtfChamber(tp_ilink);
    hit.setEmtfSegment(tp_segment_id);
    hit.setEmtfPhi(emtf_phi_w1);
    hit.setEmtfBend(emtf_bend_w1);
    hit.setEmtfTheta1(emtf_theta_w1);
    hit.setEmtfTheta2(emtf_theta_w2);
    hit.setEmtfQual1(emtf_qual_w1);
    hit.setEmtfQual2(emtf_qual_w2);
    hit.setEmtfSite(emtf_site_w1);
    hit.setEmtfHost(emtf_host_w1);
    hit.setEmtfZones(emtf_zones_w1);

    hit.setFlagNeighbor(tp_flag_neighbor);
    hit.setFlagSubstitute(tp_flag_substitute);
    hit.setFlagValid(tp_flag_valid);
}

