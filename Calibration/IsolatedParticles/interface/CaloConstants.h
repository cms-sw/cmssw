#ifndef CalibrationIsolatedParticlesCaloConstants_h
#define CalibrationIsolatedParticlesCaloConstants_h

#include <cmath>

namespace spr {

  static const double deltaEta = 0.087;   //Tower size
  static const double zFrontEE = 319.2;   //Front of EE
  static const double rFrontEB = 129.4;   //Front of EB
  static const double zFrontES = 303.2;   //Front of ES
  static const double etaBEEcal = 1.479;  //Transition eta between EB and EE
  static const double zFrontHE = 402.7;   //Front of HE
  static const double rFrontHB = 180.7;   //Front of HB
  static const double etaBEHcal = 1.392;  //Transition eta between HB and HE
  static const double zBackHE = 549.3;    //Back of HE
  static const double rBackHB = 288.8;    //Back of HB
  static const double rFrontHO = 384.8;   //Front of HO
  static const double zFrontHF = 1115.;   //Front of HF
  static const double zFrontTE = 110.0;   //Front of Tracker endcap
  static const double zBackTE = 290.0;    //Back of Tracker endcap
  static const double rBackTB = 109.0;    //Back of Tracker barrel
  static const double etaBETrak = 1.705;  //Transition eta between barrel-endcap of Tracker
}  // namespace spr
#endif
