#include "L1Trigger/GlobalCaloTrigger/interface/L1GctMet.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHtMissLut.h"

#include <cmath>

L1GctMet::L1GctMet(const unsigned ex, const unsigned ey, const L1GctMet::metAlgoType algo)
    : m_exComponent(ex), m_eyComponent(ey), m_algoType(algo), m_bitShift(0), m_htMissLut(new L1GctHtMissLut()) {}

L1GctMet::L1GctMet(const etComponentType& ex, const etComponentType& ey, const metAlgoType algo)
    : m_exComponent(ex), m_eyComponent(ey), m_algoType(algo), m_bitShift(0), m_htMissLut(new L1GctHtMissLut()) {}

L1GctMet::~L1GctMet() {}

// the Etmiss algorithm - external entry point
L1GctMet::etmiss_vec L1GctMet::metVector() const {
  etmiss_vec result;
  etmiss_internal algoResult;
  switch (m_algoType) {
    case cordicTranslate:
      algoResult = cordicTranslateAlgo(
          m_exComponent.value(), m_eyComponent.value(), (m_exComponent.overFlow() || m_eyComponent.overFlow()));
      break;

    case useHtMissLut:
      algoResult = useHtMissLutAlgo(
          m_exComponent.value(), m_eyComponent.value(), (m_exComponent.overFlow() || m_eyComponent.overFlow()));
      break;

    case oldGct:
      algoResult = oldGctAlgo(m_exComponent.value(), m_eyComponent.value());
      break;

    case floatingPoint:
      algoResult = floatingPointAlgo(m_exComponent.value(), m_eyComponent.value());
      break;

    default:
      algoResult.mag = 0;
      algoResult.phi = 0;
      break;
  }

  // The parameter m_bitShift allows us to discard additional LSB
  // in order to change the output scale.
  result.mag.setValue(algoResult.mag >> (m_bitShift));
  result.phi.setValue(algoResult.phi);

  result.mag.setOverFlow(result.mag.overFlow() || inputOverFlow());

  return result;
}

void L1GctMet::setExComponent(const unsigned ex) {
  etComponentType temp(ex);
  setExComponent(temp);
}

void L1GctMet::setEyComponent(const unsigned ey) {
  etComponentType temp(ey);
  setEyComponent(temp);
}

// private member functions - the different algorithms:

L1GctMet::etmiss_internal L1GctMet::cordicTranslateAlgo(const int ex, const int ey, const bool of) const {
  //---------------------------------------------------------------------------------
  //
  // This is an implementation of the CORDIC algorithm (COordinate Rotation for DIgital Computers)
  //
  // Starting from an initial two-component vector ex, ey, we perform successively smaller rotations
  // to transform the y component to zero. At the end of the procedure, the x component is the magnitude
  // of the original vector, scaled by a known constant factor. The azimuth angle phi is the sum of the
  // rotations applied.
  //
  // The algorithm can be used in a number of different variants for calculation of trigonometric
  // and hyperbolic functions as well as exponentials, logarithms and square roots. This variant
  // is called the "vector translation" mode in the Xilinx documentation.
  //
  // Original references:
  // Volder, J., "The CORDIC Trigonometric Computing Technique" IRE Trans. Electronic Computing, Vol.
  // EC-8, Sept. 1959, pp330-334
  // Walther, J.S., "A Unified Algorithm for Elementary Functions," Spring Joint computer conf., 1971,
  // proc., pp379-385
  //
  // Other information sources: http://www.xilinx.com/support/documentation/ip_documentation/cordic.pdf;
  // http://www.fpga-guru.com/files/crdcsrvy.pdf; and http://en.wikipedia.org/wiki/CORDIC
  //
  //---------------------------------------------------------------------------------

  etmiss_internal result;

  static const int of_val = 0x1FFF;  // set components to 8191 (decimal) if there's an overflow on the input

  static const int n_iterations = 6;
  // The angle units here are 1/32 of a 5 degree bin.
  // So a 90 degree rotation is 32*18=576 or 240 hex.
  const int cordic_angles[n_iterations] = {0x120, 0x0AA, 0x05A, 0x02E, 0x017, 0x00B};
  const int cordic_starting_angle_090 = 0x240;
  const int cordic_starting_angle_270 = 0x6C0;
  const int cordic_angle_360 = 0x900;

  const int cordic_scale_factor = 0x26E;  // decimal 622

  int x, y;
  int dx, dy;
  int z;

  if (of) {
    x = of_val;
    y = -of_val;
    z = cordic_starting_angle_090;
  } else {
    if (ey >= 0) {
      x = ey;
      y = -ex;
      z = cordic_starting_angle_090;
    } else {
      x = -ey;
      y = ex;
      z = cordic_starting_angle_270;
    }
  }

  for (int i = 0; i < n_iterations; i++) {
    dx = cordicShiftAndRoundBits(y, i);
    dy = cordicShiftAndRoundBits(x, i);
    if (y >= 0) {
      x = x + dx;
      y = y - dy;
      z = z + cordic_angles[i];
    } else {
      x = x - dx;
      y = y + dy;
      z = z - cordic_angles[i];
    }
  }

  int scaled_magnitude = x * cordic_scale_factor;
  int adjusted_angle = ((z < 0) ? (z + cordic_angle_360) : z) % cordic_angle_360;
  result.mag = scaled_magnitude >> 10;
  result.phi = adjusted_angle >> 5;
  if (result.mag > (unsigned)of_val)
    result.mag = (unsigned)of_val;
  return result;
}

int L1GctMet::cordicShiftAndRoundBits(const int e, const unsigned nBits) const {
  int r;
  if (nBits == 0) {
    r = e;
  } else {
    r = (((e >> (nBits - 1)) + 1) >> 1);
  }
  return r;
}

L1GctMet::etmiss_internal L1GctMet::useHtMissLutAlgo(const int ex, const int ey, const bool of) const {
  // The firmware discards the LSB of the input values, before forming
  // the address for the LUT. We do the same here.
  static const int maxComponent = 1 << L1GctHtMissLut::kHxOrHyMissComponentNBits;
  static const int componentMask = maxComponent - 1;
  static const int maxPosComponent = componentMask >> 1;

  static const int maxInput = 1 << (L1GctHtMissLut::kHxOrHyMissComponentNBits + kExOrEyMissComponentShift - 1);

  static const unsigned resultMagMask = (1 << L1GctHtMissLut::kHtMissMagnitudeNBits) - 1;
  static const unsigned resultPhiMask = (1 << L1GctHtMissLut::kHtMissAngleNBits) - 1;

  etmiss_internal result;

  if (m_htMissLut == nullptr) {
    result.mag = 0;
    result.phi = 0;

  } else {
    // Extract the bit fields of the input components to be used for the LUT address
    int hxCompBits = (ex >> kExOrEyMissComponentShift) & componentMask;
    int hyCompBits = (ey >> kExOrEyMissComponentShift) & componentMask;

    if (of || (abs(ex) >= maxInput) || (abs(ey) >= maxInput)) {
      hxCompBits = maxPosComponent;
      hyCompBits = maxPosComponent;
    }

    // Perform the table lookup to get the missing Ht magnitude and phi
    uint16_t lutAddress = static_cast<uint16_t>((hxCompBits << L1GctHtMissLut::kHxOrHyMissComponentNBits) | hyCompBits);

    uint16_t lutData = m_htMissLut->lutValue(lutAddress);

    result.mag = static_cast<unsigned>(lutData >> L1GctHtMissLut::kHtMissAngleNBits) & resultMagMask;
    result.phi = static_cast<unsigned>(lutData) & resultPhiMask;
  }

  return result;
}

L1GctMet::etmiss_internal L1GctMet::oldGctAlgo(const int ex, const int ey) const {
  //---------------------------------------------------------------------------------
  //
  // Calculates magnitude and direction of missing Et, given measured Ex and Ey.
  //
  // The algorithm used is suitable for implementation in hardware, using integer
  // multiplication, addition and comparison and bit shifting operations.
  //
  // Proceed in two stages. The first stage gives a result that lies between
  // 92% and 100% of the true Et, with the direction measured in 45 degree bins.
  // The final precision depends on the number of factors used in corrFact.
  // The present version with eleven factors gives a precision of 1% on Et, and
  // finds the direction to the nearest 5 degrees.
  //
  //---------------------------------------------------------------------------------
  etmiss_internal result;

  unsigned eneCoarse, phiCoarse;
  unsigned eneCorect, phiCorect;

  const unsigned root2fact = 181;
  const unsigned corrFact[11] = {24, 39, 51, 60, 69, 77, 83, 89, 95, 101, 106};
  const unsigned corrDphi[11] = {0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4};

  std::vector<bool> s(3);
  unsigned Mx, My, Mw;

  unsigned Dx, Dy;
  unsigned eFact;

  unsigned b, phibin;
  bool midphi = false;

  // Here's the coarse calculation, with just one multiply operation
  //
  My = static_cast<unsigned>(abs(ey));
  Mx = static_cast<unsigned>(abs(ex));
  Mw = (((Mx + My) * root2fact) + 0x80) >> 8;

  s.at(0) = (ey < 0);
  s.at(1) = (ex < 0);
  s.at(2) = (My > Mx);

  phibin = 0;
  b = 0;
  for (int i = 0; i < 3; i++) {
    if (s.at(i)) {
      b = 1 - b;
    }
    phibin = 2 * phibin + b;
  }

  eneCoarse = std::max(std::max(Mx, My), Mw);
  phiCoarse = phibin * 9;

  // For the fine calculation we multiply both input components
  // by all the factors in the corrFact list in order to find
  // the required corrections to the energy and angle
  //
  for (eFact = 0; eFact < 10; eFact++) {
    Dx = (Mx * corrFact[eFact]) >> 8;
    Dy = (My * corrFact[eFact]) >> 8;
    if ((Dx >= My) || (Dy >= Mx)) {
      midphi = false;
      break;
    }
    if ((Mx + Dx) >= (My - Dy) && (My + Dy) >= (Mx - Dx)) {
      midphi = true;
      break;
    }
  }
  eneCorect = (eneCoarse * (128 + eFact)) >> 7;
  if (midphi ^ (b == 1)) {
    phiCorect = phiCoarse + 8 - corrDphi[eFact];
  } else {
    phiCorect = phiCoarse + corrDphi[eFact];
  }

  // Store the result of the calculation
  //
  result.mag = eneCorect;
  result.phi = phiCorect;

  return result;
}

L1GctMet::etmiss_internal L1GctMet::floatingPointAlgo(const int ex, const int ey) const {
  etmiss_internal result;

  double fx = static_cast<double>(ex);
  double fy = static_cast<double>(ey);
  double fmag = sqrt(fx * fx + fy * fy);
  double fphi = 36. * atan2(fy, fx) / M_PI;

  result.mag = static_cast<unsigned>(fmag);
  if (fphi >= 0) {
    result.phi = static_cast<unsigned>(fphi);
  } else {
    result.phi = static_cast<unsigned>(fphi + 72.);
  }

  return result;
}

void L1GctMet::setEtScale(const L1CaloEtScale* const fn) { m_htMissLut->setEtScale(fn); }

void L1GctMet::setEtComponentLsb(const double lsb) {
  m_htMissLut->setExEyLsb(lsb * static_cast<double>(1 << kExOrEyMissComponentShift));
}

const L1CaloEtScale* L1GctMet::etScale() const { return m_htMissLut->etScale(); }

const double L1GctMet::componentLsb() const { return m_htMissLut->componentLsb(); }

/// Private method to check for an overflow condition on the input components
/// Allows the check to depend on the algorithm type
const bool L1GctMet::inputOverFlow() const {
  bool result = m_exComponent.overFlow() || m_eyComponent.overFlow();

  if (m_algoType == useHtMissLut) {
    static const int maxComponentInput =
        (1 << (L1GctHtMissLut::kHxOrHyMissComponentNBits + kExOrEyMissComponentShift - 1)) - 1;

    // Emulate the (symmetric) overflow condition used in the firmware
    result |= (m_exComponent.value() > maxComponentInput) || (m_exComponent.value() < -maxComponentInput) ||
              (m_eyComponent.value() > maxComponentInput) || (m_eyComponent.value() < -maxComponentInput);
  }

  return result;
}
