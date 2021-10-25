#ifndef L1TMuonEndCap_PrimitiveConversion_h
#define L1TMuonEndCap_PrimitiveConversion_h

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"

class SectorProcessorLUT;

class PrimitiveConversion {
public:
  void configure(const GeometryTranslator* tp_geom,
                 const SectorProcessorLUT* pc_lut,
                 int verbose,
                 int endcap,
                 int sector,
                 int bx,
                 int bxShiftCSC,
                 int bxShiftRPC,
                 int bxShiftGEM,
                 int bxShiftME0,
                 const std::vector<int>& zoneBoundaries,
                 int zoneOverlap,
                 bool duplicateTheta,
                 bool fixZonePhi,
                 bool useNewZones,
                 bool fixME11Edges,
                 bool bugME11Dupes,
                 bool useRun3CCLUT);

  void process(const std::map<int, TriggerPrimitiveCollection>& selected_prim_map, EMTFHitCollection& conv_hits) const;

  const SectorProcessorLUT& pc_lut() const { return *pc_lut_; }

  // CSC functions
  void convert_csc(int pc_sector,
                   int pc_station,
                   int pc_chamber,
                   int pc_segment,
                   const TriggerPrimitive& muon_primitive,
                   EMTFHit& conv_hit) const;

  void convert_csc_details(EMTFHit& conv_hit) const;  // with specific firmware impl

  // RPC functions
  void convert_rpc(int pc_sector,
                   int pc_station,
                   int pc_chamber,
                   int pc_segment,
                   const TriggerPrimitive& muon_primitive,
                   EMTFHit& conv_hit) const;

  void convert_rpc_details(EMTFHit& conv_hit, bool isCPPF) const;  // with specific firmware impl

  // GEM functions
  void convert_gem(int pc_sector,
                   int pc_station,
                   int pc_chamber,
                   int pc_segment,
                   const TriggerPrimitive& muon_primitive,
                   EMTFHit& conv_hit) const;

  void convert_other_details(EMTFHit& conv_hit) const;  // no firmware impl

  // ME0 functions
  void convert_me0(int pc_sector,
                   int pc_station,
                   int pc_chamber,
                   int pc_segment,
                   const TriggerPrimitive& muon_primitive,
                   EMTFHit& conv_hit) const;

  // DT functions
  void convert_dt(int pc_sector,
                  int pc_station,
                  int pc_chamber,
                  int pc_segment,
                  const TriggerPrimitive& muon_primitive,
                  EMTFHit& conv_hit) const;

  // Aux functions
  int get_zone_code(const EMTFHit& conv_hit, int th) const;

  int get_phzvl(const EMTFHit& conv_hit, int zone_code) const;

  int get_fs_zone_code(const EMTFHit& conv_hit) const;

  int get_fs_segment(const EMTFHit& conv_hit, int fw_station, int fw_cscid, int pc_segment) const;

  int get_bt_station(const EMTFHit& conv_hit, int fw_station, int fw_cscid, int pc_segment) const;

  int get_bt_segment(const EMTFHit& conv_hit, int fw_station, int fw_cscid, int pc_segment) const;

  bool is_valid_for_run2(const EMTFHit& conv_hit) const;

private:
  const GeometryTranslator* tp_geom_;

  const SectorProcessorLUT* pc_lut_;

  int verbose_, endcap_, sector_, bx_;

  int bxShiftCSC_, bxShiftRPC_, bxShiftGEM_, bxShiftME0_;

  std::vector<int> zoneBoundaries_;
  int zoneOverlap_;
  bool duplicateTheta_, fixZonePhi_, useNewZones_, fixME11Edges_;
  bool bugME11Dupes_;
  // Run 3 CCLUT algorithm
  bool useRun3CCLUT_;
};

#endif
