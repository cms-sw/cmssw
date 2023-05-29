#ifndef L1Trigger_L1TTrackMatch_Cordic_HH
#define L1Trigger_L1TTrackMatch_Cordic_HH

#include "L1Trigger/L1TTrackMatch/interface/L1TkEtMissEmuAlgo.h"

/*
** class  : Cordic
** author : Christopher Brown
** date   : 19/02/2021
** brief  : Integer sqrt and atan calculation for TrackMET emulation

**        : 
*/

class Cordic {
public:
  Cordic();
  Cordic(const int aSteps, bool debug);

  template <typename T>
  void cordic_subfunc(T &x, T &y, T &z) const;
  l1tmetemu::EtMiss toPolar(l1tmetemu::Et_t x, l1tmetemu::Et_t y) const;

private:
  const int cordicSteps;

  const bool debug;

  // To calculate atan
  std::vector<l1tmetemu::atan_lut_fixed_t> atanLUT;
  // To normalise final magnitude
  std::vector<l1tmetemu::atan_lut_fixed_t> magNormalisationLUT;
};

#endif
