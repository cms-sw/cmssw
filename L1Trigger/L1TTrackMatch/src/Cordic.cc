#include "L1Trigger/L1TTrackMatch/interface/Cordic.h"

#include <cmath>
#include <memory>

using namespace l1tmetemu;

Cordic::Cordic(const int aSteps, bool debug)
    : cordicSteps(aSteps),
      debug(debug) {
  atanLUT.reserve(aSteps);
  magNormalisationLUT.reserve(aSteps);

  if (debug) {
    edm::LogVerbatim("L1TkEtMissEmulator") << "=====atan LUT=====";
  }

  for (int i = 0; i < aSteps; i++) {
    atanLUT.push_back(E2t_t(atan(pow(2, -i))));
    if (debug) {
      edm::LogVerbatim("L1TkEtMissEmulator") << atanLUT[i] << " | ";
    }
  }
  if (debug) {
    edm::LogVerbatim("L1TkEtMissEmulator") << "\n=====Normalisation LUT=====";
  }

  float val = 1.0;
  for (int j = 0; j < aSteps; j++) {
    val = val / (pow(1 + pow(4, -j), 0.5));
    magNormalisationLUT.push_back(E2t_t(val));
    if (debug) {
      edm::LogVerbatim("L1TkEtMissEmulator") << magNormalisationLUT[j] << " | ";
    }
  }
}

EtMiss Cordic::toPolar(Et_t x, Et_t y) const {
  E2t_t in_x = x;
  E2t_t in_y = y;
  E2t_t new_x = 0;
  E2t_t new_y = 0;

  E2t_t phi = 0;
  E2t_t new_phi = 0;
  bool sign = false;

  EtMiss ret_etmiss;

  if (debug) {
    edm::LogVerbatim("L1TkEtMissEmulator") << "\n=====Cordic Steps=====";
  }

  if (in_x >= 0 && in_y >= 0) {
    phi = E2t_t(M_PI);
    sign = true;

  } else if (in_x < 0 && in_y >= 0) {
    phi = E2t_t(2*M_PI);
    sign = false;
    in_x = -in_x;

  } else if (in_x < 0 && in_y < 0) {
    phi = 0;
    sign = true;
    in_x = -in_x;
    in_y = -in_y;

  } else {
    phi = E2t_t(M_PI);
    sign = false;
    in_y = -in_y;
  }

  for (int step = 0; step < cordicSteps; step++) {
    if (in_y < 0) {
      new_x = in_x - (in_y >> step);
      new_y = in_y + (in_x >> step);
    } else {
      new_x = in_x + (in_y >> step);
      new_y = in_y - (in_x >> step);
    }

    if ((in_y < 0) == sign) {
      new_phi = phi - atanLUT[step];
    } else {
      new_phi = phi + atanLUT[step];
    }

    in_x = new_x;
    in_y = new_y;
    phi = new_phi;

    if (debug) {
      edm::LogVerbatim("L1TkEtMissEmulator")
          << " Cordic x: " << in_x << " Cordic y: " << in_y << " Cordic phi: " << phi << "\n";
    }
  }

  // Cordic performs calculation in internal Et granularity, convert to final
  // granularity for Et word

  E2t_t tempMET = in_x * magNormalisationLUT[cordicSteps - 1];
  ret_etmiss.Et = tempMET;
  ret_etmiss.Phi = phi.to_double() / kStepMETwordPhi;
  return ret_etmiss;
}
