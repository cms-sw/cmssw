// Trigger Primitive Converter
//
// Takes in raw information from the TriggerPrimitive class(part of L1TMuon software package);
// and outputs vector of 'ConvertedHits'
//

#include "L1Trigger/L1TMuonEndCap/interface/EmulatorClasses.h"

class PrimitiveConverter {
 public:
  PrimitiveConverter();
  std::vector<ConvertedHit> convert(std::vector<TriggerPrimitive> TriggPrim, int SectIndex);

 private:
  // don't mind the magid numbers here, this is throw-away code pending CondFormat update in works:
  unsigned int Ph_Disp_Neighbor_[12][61];
  int Ph_Init_Neighbor_[12][5][16];
  int Th_Corr_Neighbor_[2][12][4][96];
  unsigned int Th_Init_Neighbor_[12][61];
  int Th_LUT_St1_Neighbor_[2][12][16][64];
  int Th_LUT_St234_Neighbor_[3][12][11][112];
};
