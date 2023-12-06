#ifndef L1Trigger_L1TMuonEndCapPhase2_EMTFTypes_h
#define L1Trigger_L1TMuonEndCapPhase2_EMTFTypes_h

#include<array>
#include<vector>

#include "ap_int.h"
#include "ap_fixed.h"
#include "DataFormats/L1TMuonPhase2/interface/EMTFHit.h"
#include "DataFormats/L1TMuonPhase2/interface/EMTFTrack.h"
#include "DataFormats/L1TMuonPhase2/interface/EMTFInput.h"
#include "L1Trigger/L1TMuon/interface/GeometryTranslator.h"
#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitive.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFConstants.h"

namespace emtf::phase2 {

    // Trigger Primitives
    typedef L1TMuon::TriggerPrimitive TriggerPrimitive;
    typedef std::vector<TPEntry> TPCollection;
    typedef std::map<int, TPCollection> BXTPCMap;
    typedef std::map<int, TPCollection> ILinkTPCMap;

    // Hits
    typedef l1t::phase2::EMTFHit EMTFHit;
    typedef l1t::phase2::EMTFHitCollection EMTFHitCollection;

    // Tracks
    typedef l1t::phase2::EMTFTrack EMTFTrack;
    typedef l1t::phase2::EMTFTrackCollection EMTFTrackCollection;

    // Inputs
    typedef l1t::phase2::EMTFInput EMTFInput;
    typedef l1t::phase2::EMTFInputCollection EMTFInputCollection;

    // General
    typedef ap_uint<1> flag_t;

    // Segments
    enum class site_id_t {
        kME11 = 0, kME12 = 1 , kME2 = 2 , kME3 = 3, kME4 = 4, 
        kRE1  = 5, kRE2  = 6 , kRE3 = 7 , kRE4 = 8, 
        kGE11 = 9, kGE21 = 10, kME0 = 11,
        size, begin = 0, end=size
    };

    enum class feature_id_t {
        kPhi = 0, kTheta = 1 , kBend = 2 , kQuality = 3, kTime = 4, 
        size, begin = 0, end=size 
    };

    enum class theta_id_t {
        kTheta1 = 0, kTheta2 = 1,
        size, begin = 0, end=size 
    };
    
    enum class reduced_site_id_t {
        kME1 = 0, kME2 = 1,
        kME3 = 2, kME4 = 3,
        kME0 = 4,
        size, begin = 0, end=size 
    };

    typedef ap_uint <13> seg_phi_t   ;
    typedef ap_int  <10> seg_bend_t  ;
    typedef ap_uint <8 > seg_theta_t ;
    typedef ap_uint <4 > seg_qual_t  ;
    typedef ap_int  <4 > seg_time_t  ;
    typedef ap_uint <3 > seg_zones_t ;
    typedef ap_uint <3 > seg_tzones_t;
    typedef ap_uint <1 > seg_cscfr_t ;
    typedef ap_uint <1 > seg_layer_t ;
    typedef ap_int  <2 > seg_bx_t    ;
    typedef ap_uint <1 > seg_valid_t ;

    struct segment_t {
        seg_phi_t    phi   ;
        seg_bend_t   bend  ;
        seg_theta_t  theta1;
        seg_theta_t  theta2;
        seg_qual_t   qual1 ;
        seg_qual_t   qual2 ;
        seg_time_t   time  ;
        seg_zones_t  zones ;
        seg_tzones_t tzones;
        seg_cscfr_t  cscfr ;
        seg_layer_t  layer ;
        seg_bx_t     bx    ;
        seg_valid_t  valid ;
    };

    typedef std::array<segment_t, v3::kNumSegments> segment_collection_t;

    // Tracks
    typedef ap_uint <2 > trk_zone_t      ;
    typedef ap_uint <2 > trk_tzone_t     ;
    typedef ap_uint <9 > trk_col_t       ;
    typedef ap_uint <3 > trk_patt_t      ;
    typedef ap_uint <6 > trk_qual_t      ;
    typedef ap_uint <2 > trk_gate_t      ;
    typedef ap_uint <1 > trk_q_t         ;
    typedef ap_uint <13> trk_pt_t        ;
    typedef ap_uint <7 > trk_rels_t      ;
    typedef ap_int  <7 > trk_dxy_t       ;
    typedef ap_int  <5 > trk_z0_t        ;
    typedef ap_int  <13> trk_phi_t       ;
    typedef ap_int  <13> trk_eta_t       ;
    typedef ap_uint <4 > trk_beta_t      ;
    typedef ap_uint <1 > trk_valid_t     ;
    typedef ap_uint <8 > trk_site_seg_t  ;
    typedef ap_uint <1 > trk_site_bit_t  ;
    typedef ap_int  <13> trk_feature_t   ;
    typedef ap_int  <10> trk_nn_address_t;

    struct track_t {
        typedef std::array<trk_site_seg_t, v3::kNumTrackSites>    site_segs_t;
        typedef std::array<trk_site_bit_t, v3::kNumTrackSites>    site_mask_t;
        typedef std::array<trk_feature_t , v3::kNumTrackFeatures> features_t;

        trk_zone_t  zone             ;
        trk_col_t   col              ;
        trk_patt_t  pattern          ;
        trk_qual_t  quality          ;
        trk_q_t     q                ;
        trk_pt_t    pt               ;
        trk_rels_t  rels             ;
        trk_dxy_t   dxy              ;
        trk_z0_t    z0               ;
        seg_phi_t   phi              ;
        seg_theta_t theta            ;
        trk_eta_t   eta              ;
        trk_beta_t  beta             ;
        trk_valid_t valid            ;
        site_segs_t site_segs        ;
        site_mask_t site_mask        ;
        site_mask_t site_rm_mask     ;
        features_t  features         ;
        trk_nn_address_t pt_address  ;
        trk_nn_address_t rels_address;
        trk_nn_address_t dxy_address ;
    };

    // Hitmaps
    typedef ap_uint<v3::kHitmapNCols> hitmap_row_t;
    typedef std::array<hitmap_row_t, v3::kHitmapNRows> hitmap_t;

    // Roads
    struct road_t {
        trk_zone_t zone   ;
        trk_col_t  col    ;
        trk_patt_t pattern;
        trk_qual_t quality;
    };

    typedef std::array<road_t, v3::kHitmapNCols> road_collection_t;

    // Reduced Track
    struct reduced_track_t {
        typedef std::array<trk_site_seg_t, v3::kNumTrackSitesRM> site_segs_t;
        typedef std::array<trk_site_bit_t, v3::kNumTrackSitesRM> site_mask_t;

        trk_valid_t valid;
        site_segs_t site_segs;
        site_mask_t site_mask;
    };
}

typedef L1TMuon::subsystem_type SubsystemType;
typedef L1TMuon::GeometryTranslator GeometryTranslator;

typedef L1TMuon::TriggerPrimitive::DTData DTData;
typedef L1TMuon::TriggerPrimitive::CSCData CSCData;
typedef L1TMuon::TriggerPrimitive::RPCData RPCData;
typedef L1TMuon::TriggerPrimitive::GEMData GEMData;
typedef L1TMuon::TriggerPrimitive::ME0Data ME0Data;

#endif  // L1Trigger_L1TMuonEndCapPhase2_EMTFTypes_h
