#include "DataFormats/L1TMuonPhase2/interface/EMTFTrack.h"

using namespace l1t::phase2;

EMTFTrack::EMTFTrack()
    : endcap_(0),
      sector_(0),
      bx_(0),
      unconstrained_(false),
      valid_(false),
      model_pt_address_(0),
      model_dxy_address_(0),
      model_pattern_(0),
      model_qual_(0),
      model_phi_(0),
      model_eta_(0),
      model_features_{},
      emtf_q_(0),
      emtf_pt_(0),
      emtf_d0_(0),
      emtf_z0_(0),
      emtf_beta_(0),
      emtf_mode_v1_(0),
      emtf_mode_v2_(0),
      site_hits_{},
      site_segs_{},
      site_mask_{},
      site_rm_mask_{} {
  // Do Nothing
}
