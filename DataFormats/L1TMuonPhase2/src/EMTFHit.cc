#include "DataFormats/L1TMuonPhase2/interface/EMTFHit.h"

using namespace l1t::phase2;

EMTFHit::EMTFHit()
    : id_(0),

      raw_det_id_(0),
      subsystem_(0),
      endcap_(0),
      sector_(0),
      subsector_(0),
      station_(0),
      ring_(0),
      roll_(0),
      layer_(0),
      chamber_(0),

      csc_id_(0),
      csc_fr_(0),

      strip_(0),
      strip_lo_(0),
      strip_hi_(0),
      strip_quart_(0),       // Run 3
      strip_eighth_(0),      // Run 3
      strip_quart_bit_(0),   // Run 3
      strip_eighth_bit_(0),  // Run 3

      wire1_(0),
      wire2_(0),

      bx_(0),
      subbx_(0),

      quality_(0),
      pattern_(0),

      glob_phi_(0),
      glob_theta_(0),
      glob_perp_(0),
      glob_z_(0),
      glob_time_(0),

      emtf_chamber_(0),
      emtf_segment_(0),
      emtf_phi_(0),
      emtf_bend_(0),
      emtf_slope_(0),
      emtf_theta1_(0),
      emtf_theta2_(0),
      emtf_qual1_(0),
      emtf_qual2_(0),
      emtf_time_(0),
      emtf_site_(0),
      emtf_host_(0),
      emtf_zones_(0),
      emtf_timezones_(0),

      flag_neighbor_(false),
      flag_substitute_(false),
      flag_valid_(false) {
  // Do Nothing
}
