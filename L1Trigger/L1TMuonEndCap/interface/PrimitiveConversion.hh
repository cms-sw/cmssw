#ifndef L1TMuonEndCap_PrimitiveConversion_hh
#define L1TMuonEndCap_PrimitiveConversion_hh

#include "L1Trigger/L1TMuonEndCap/interface/Common.hh"


class SectorProcessorLUT;

class PrimitiveConversion {
public:
  void configure(
      const GeometryTranslator* tp_geom,
      const SectorProcessorLUT* lut,
      int verbose, int endcap, int sector, int bx,
      int bxShiftCSC, int bxShiftRPC,
      const std::vector<int>& zoneBoundaries, int zoneOverlap, int zoneOverlapRPC,
      bool duplicateTheta, bool fixZonePhi, bool useNewZones, bool fixME11Edges,
      bool bugME11Dupes
  );

  template<typename T>
  void process(
      T tag,
      const std::map<int, TriggerPrimitiveCollection>& selected_prim_map,
      EMTFHitCollection& conv_hits
  ) const;

  const SectorProcessorLUT& lut() const;

  // CSC functions
  void convert_csc(
      int pc_sector, int pc_station, int pc_chamber, int pc_segment,
      const TriggerPrimitive& muon_primitive,
      EMTFHit& conv_hit
  ) const;
  void convert_csc_details(EMTFHit& conv_hit) const;

  // RPC functions
  void convert_rpc(
      int pc_sector, int pc_station, int pc_chamber, int pc_segment,
      const TriggerPrimitive& muon_primitive,
      EMTFHit& conv_hit
  ) const;
  void convert_rpc_details(EMTFHit& conv_hit) const;

private:
  const GeometryTranslator* tp_geom_;

  const SectorProcessorLUT* lut_;

  int verbose_, endcap_, sector_, bx_;

  int bxShiftCSC_, bxShiftRPC_;

  std::vector<int> zoneBoundaries_;
  int zoneOverlap_, zoneOverlapRPC_;
  bool duplicateTheta_, fixZonePhi_, useNewZones_, fixME11Edges_;
  bool bugME11Dupes_;
};

#endif

