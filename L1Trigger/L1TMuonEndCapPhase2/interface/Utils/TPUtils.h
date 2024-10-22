#ifndef L1Trigger_L1TMuonEndCapPhase2_TPUtils_h
#define L1Trigger_L1TMuonEndCapPhase2_TPUtils_h

namespace emtf::phase2::tp {

  // _______________________________________________________________________
  // radians <-> degrees
  float degToRad(float deg);

  float radToDeg(float rad);

  // _______________________________________________________________________
  // phi range: [-180..180] or [-pi..pi]
  float wrapPhiDeg(float);

  float wrapPhiRad(float);

  // _______________________________________________________________________
  // theta
  float calcThetaRadFromEta(float);

  float calcThetaDegFromEta(float);

  float calcThetaRadFromInt(int);

  float calcThetaDegFromInt(int);

  int calcThetaInt(int, float);

  // _______________________________________________________________________
  // phi
  float calcPhiGlobDegFromLoc(int, float);

  float calcPhiGlobRadFromLoc(int, float);

  float calcPhiLocDegFromInt(int);

  float calcPhiLocRadFromInt(int);

  float calcPhiLocDegFromGlob(int, float);

  int calcPhiInt(int, float);

}  // namespace emtf::phase2::tp

#endif  // namespace L1Trigger_L1TMuonEndCapPhase2_TPUtils_h
