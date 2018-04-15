#ifndef L1TMuonEndCap_TTPrimitiveConversion_h
#define L1TMuonEndCap_TTPrimitiveConversion_h

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"


class SectorProcessorLUT;

class TTPrimitiveConversion {
public:
  void configure(
      const TTGeometryTranslator* tp_ttgeom,
      const SectorProcessorLUT* lut,
      int verbose, int endcap, int sector, int bx
  );

  void process(
      const std::map<int, TTTriggerPrimitiveCollection>& selected_ttprim_map,
      EMTFHitCollection& conv_hits
  ) const;

  void process_no_prim_sel(
      const TTTriggerPrimitiveCollection& ttmuon_primitives,
      EMTFHitCollection& conv_hits
  ) const;

  const SectorProcessorLUT& lut() const { return *lut_; }

  // TT functions
  void convert_tt(
      const TTTriggerPrimitive& ttmuon_primitive,
      EMTFHit& conv_hit
  ) const;

private:
  const TTGeometryTranslator* tp_ttgeom_;

  const SectorProcessorLUT* lut_;

  int verbose_, endcap_, sector_, bx_;
};

#endif

