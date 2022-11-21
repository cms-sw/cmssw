#include "L1Trigger/L1TTrackMatch/interface/Cordic.h"

#include <cmath>
#include <iomanip>
#include <memory>

using namespace l1tmetemu;

Cordic::Cordic(const int aSteps, bool debug) : cordicSteps(aSteps), debug(debug) {
  atanLUT.reserve(aSteps);
  magNormalisationLUT.reserve(aSteps);

  if (debug) {
    edm::LogVerbatim("L1TkEtMissEmulator") << "=====atan LUT=====";
  }

  for (int i = 0; i < aSteps; i++) {
    atanLUT.push_back(l1tmetemu::atan_lut_fixed_t(atan(pow(2, -i))));
    if (debug) {
      edm::LogVerbatim("L1TkEtMissEmulator") << atanLUT[i] << " | ";
    }
  }
  if (debug) {
    edm::LogVerbatim("L1TkEtMissEmulator") << "\n=====Normalisation LUT=====";
  }

  double val = 1.0;
  for (int j = 0; j < aSteps; j++) {
    val = val / (pow(1 + pow(4, -j), 0.5));
    magNormalisationLUT.push_back(l1tmetemu::atan_lut_fixed_t(val));
    if (debug) {
      edm::LogVerbatim("L1TkEtMissEmulator") << magNormalisationLUT[j] << " | ";
    }
  }
}

template <typename T>
void Cordic::cordic_subfunc(T &x, T &y, T &z) const {
  if (debug) {
    edm::LogVerbatim("L1TkEtMissEmulator") << "\n=====Cordic Initial Conditions=====\n"
                                           << "Cordic x: " << x << " Cordic y: " << y << " Cordic z: " << z << "\n"
                                           << "\n=====Cordic Steps=====";
  }

  T tx, ty, tz;

  for (int step = 0; step < cordicSteps; step++) {
    if (y < 0) {
      tx = x - (y >> step);
      ty = y + (x >> step);
      tz = z - atanLUT[step];
    } else {
      tx = x + (y >> step);
      ty = y - (x >> step);
      tz = z + atanLUT[step];
    }

    x = tx;
    y = ty;
    z = tz;

    if (debug) {
      edm::LogVerbatim("L1TkEtMissEmulator")
          << "Cordic x: " << x << " Cordic y: " << y << " Cordic phi: " << z << "\n"
          << " Cordic gain: " << magNormalisationLUT[step] << " kStepMETwordPhi: " << kStepMETwordPhi << "\n";
    }
  }
}

EtMiss Cordic::toPolar(Et_t x, Et_t y) const {
  EtMiss ret_etmiss;

  if (debug) {
    edm::LogVerbatim("L1TkEtMissEmulator") << "\n=====toPolar input=====\n"
                                           << "x: " << x << " y: " << y;
  }

  // Some needed constants
  const ap_fixed<l1tmetemu::Et_t::width + 1, 3> pi = M_PI;                   // pi
  const ap_fixed<l1tmetemu::Et_t::width + 2, 3> pi2 = M_PI / 2.;             // pi/2
  const l1tmetemu::METWordphi_t pistep = M_PI / l1tmetemu::kStepMETwordPhi;  // (pi) / l1tmetemu::kStepMETwordPhi
  const l1tmetemu::METWordphi_t pi2step = pistep / 2.;                       // (pi/2) / l1tmetemu::kStepMETwordPhi

  // Find the sign of the inputs
  ap_uint<2> signx = (x > 0) ? 2 : (x == 0) ? 1 : 0;
  ap_uint<2> signy = (y > 0) ? 2 : (y == 0) ? 1 : 0;

  // Corner cases
  if (signy == 1 && signx == 2) {  // y == 0 and x > 0
    ret_etmiss.Et = x;
    ret_etmiss.Phi = 0;
    return ret_etmiss;
  } else if (signy == 1 && signx == 0) {  // y == 0 and x < 0
    ret_etmiss.Et = -x;
    ret_etmiss.Phi = pistep;
    return ret_etmiss;
  } else if (signy == 2 && signx == 1) {  // y > 0 and x == 0
    ret_etmiss.Et = y;
    ret_etmiss.Phi = pi2step;
    return ret_etmiss;
  } else if (signy == 0 && signx == 1) {  // y < 0 and x == 0
    ret_etmiss.Et = -y;
    ret_etmiss.Phi = -pi2step;
    return ret_etmiss;
  }

  // Take absolute values to operate on the range (0, pi/2)
  ap_fixed<Et_t::width + 1, Et_t::iwidth + 1, Et_t::qmode, Et_t::omode> absx, absy;
  if (signy == 0) {
    absy = -y;
  } else {
    absy = y;
  }

  if (signx == 0) {
    absx = -x;
  } else {
    absx = x;
  }

  if (debug) {
    edm::LogVerbatim("L1TkEtMissEmulator") << "\n=====Abs input=====\n"
                                           << "abs(x): " << absx.to_double() << " abs(y): " << absy.to_double();
  }

  // Normalization (operate on a unit circle)
  ap_fixed<Et_t::width + 1, 2, Et_t::qmode, Et_t::omode> absx_sft, absy_sft;
  for (int i = 0; i < Et_t::width + 1; i++) {
    absx_sft[i] = absx[i];
    absy_sft[i] = absy[i];
  }

  if (debug) {
    edm::LogVerbatim("L1TkEtMissEmulator")
        << "\n=====Normalized input=====\n"
        << "norm(abs(x)): " << absx_sft.to_double() << " norm(abs(y)): " << absy_sft.to_double();
  }

  // Setup the CORDIC inputs/outputs
  ap_fixed<Et_t::width + 7, 3, Et_t::qmode, Et_t::omode> cx, cy, cphi;
  if (absy > absx) {
    cx = absy_sft;
    cy = absx_sft;
    cphi = 0;
  } else {
    cx = absx_sft;
    cy = absy_sft;
    cphi = 0;
  }

  if (debug) {
    edm::LogVerbatim("L1TkEtMissEmulator")
        << "\n=====CORDIC function arguments=====\n"
        << "x: " << cx.to_double() << " y: " << cy.to_double() << " phi: " << cphi.to_double();
  }

  // Perform the CORDIC (vectoring) function
  cordic_subfunc(cx, cy, cphi);

  // Reorient the outputs to their appropriate quadrant
  if (absy > absx) {
    cphi = pi2 - cphi;
  }

  ap_fixed<Et_t::width, 3, Et_t::qmode, Et_t::omode> ophi;
  if (signx == 0 && signy == 2) {  // x < 0 and y > 0
    ophi = pi - cphi;
  } else if (signx == 0 && signy == 0) {  // x < 0 and y < 0
    ophi = cphi - pi;
  } else if (signx == 2 && signy == 0) {  // x > 0 and y < 0
    ophi = -cphi;
  } else {
    ophi = cphi;
  }

  // Re-scale the outputs
  Et_t magnitude = ((ap_fixed<Et_t::width + Et_t::iwidth + 3 + 1, Et_t::iwidth, Et_t::qmode, Et_t::omode>)cx)
                   << (Et_t::iwidth + 1 - 2);
  ret_etmiss.Et = l1tmetemu::Et_t(magnitude * magNormalisationLUT[cordicSteps - 1]);
  ret_etmiss.Phi = ophi * pi_bins_fixed_t(kBinsInPi);

  if (debug) {
    edm::LogVerbatim("L1TkEtMissEmulator")
        << "\n=====toPolar output=====\n"
        << std::setprecision(8) << "magnitude: " << magnitude.to_double() << " phi: " << ophi.to_double()
        << " kBinsInPi: " << pi_bins_fixed_t(kBinsInPi).to_double() << "\n"
        << "Et: " << ret_etmiss.Et.to_double() << " phi (int): " << ret_etmiss.Phi.to_int() << "\n";
  }

  return ret_etmiss;
}
