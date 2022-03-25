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
  Cordic(int aPhiScale, int aMagnitudeBits, const int aSteps, bool debug);

  l1tmetemu::EtMiss toPolar(l1tmetemu::Et_t x, l1tmetemu::Et_t y) const;

private:
  // Scale for Phi calculation to maintain precision
  const int mPhiScale;
  // Scale for Magnitude calculation
  const int mMagnitudeScale;
  // Bit width for internal magnitude
  const int mMagnitudeBits;
  // Number of cordic iterations
  const int cordicSteps;

  const bool debug;

  // To calculate atan
  std::vector<l1tmetemu::METphi_t> atanLUT;
  // To normalise final magnitude
  std::vector<l1tmetemu::Et_t> magNormalisationLUT;
};

#endif
