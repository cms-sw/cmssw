#ifndef RecoLocalCalo_EcalDeadChannelRecoveryAlgos_EcalCrystalMatrixProbability_CC
#define RecoLocalCalo_EcalDeadChannelRecoveryAlgos_EcalCrystalMatrixProbability_CC

//
//  Original Author:   Stilianos Kesisoglou - Institute of Nuclear and Particle
// Physics NCSR Demokritos (Stilianos.Kesisoglou@cern.ch)
//          Created:   Mon Feb 04 10:45:16 EET 2013
//

#include "TMath.h"

template <typename T> class EcalCrystalMatrixProbality {
 public:
  static double Central(double x);
  static double Diagonal(double x);
  static double UpDown(double x);
  static double ReftRight(double x);
};

#endif
