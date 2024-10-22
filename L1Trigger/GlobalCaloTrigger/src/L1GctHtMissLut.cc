#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHtMissLut.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

#include <cmath>

//DEFINE STATICS
const int L1GctHtMissLut::NAddress = 2 * L1GctHtMissLut::kHxOrHyMissComponentNBits;
const int L1GctHtMissLut::NData = L1GctHtMissLut::kHtMissMagnitudeNBits + L1GctHtMissLut::kHtMissAngleNBits;

L1GctHtMissLut::L1GctHtMissLut(const L1CaloEtScale* const scale, const double lsb)
    : L1GctLut<NAddress, NData>(), m_etScale(scale), m_componentLsb(lsb) {
  if (scale != nullptr)
    m_setupOk = true;
}

L1GctHtMissLut::L1GctHtMissLut() : L1GctLut<NAddress, NData>(), m_etScale(nullptr), m_componentLsb(1.0) {}

L1GctHtMissLut::L1GctHtMissLut(const L1GctHtMissLut& lut)
    : L1GctLut<NAddress, NData>(), m_etScale(lut.etScale()), m_componentLsb(lut.componentLsb()) {
  if (m_etScale != nullptr)
    m_setupOk = true;
}

L1GctHtMissLut::~L1GctHtMissLut() {}

uint16_t L1GctHtMissLut::value(const uint16_t lutAddress) const {
  uint16_t result = 0;

  if (lutAddress != 0) {
    static const int maxComponent = 1 << kHxOrHyMissComponentNBits;
    static const int componentMask = maxComponent - 1;

    static const int magnitudeMask = (1 << kHtMissMagnitudeNBits) - 1;
    static const int angleMask = (1 << kHtMissAngleNBits) - 1;

    // Extract the bits corresponding to hx and hy components
    int hxCompGct = static_cast<int>(lutAddress >> kHxOrHyMissComponentNBits) & componentMask;
    int hyCompGct = static_cast<int>(lutAddress) & componentMask;

    // These are twos-complement integers - if the MSB is set, the value is negative
    if (hxCompGct >= maxComponent / 2)
      hxCompGct -= maxComponent;
    if (hyCompGct >= maxComponent / 2)
      hyCompGct -= maxComponent;

    // Convert to GeV. Add 0.5 to each component to compensate for truncation errors.
    double hxCompGeV = m_componentLsb * (static_cast<double>(hxCompGct) + 0.5);
    double hyCompGeV = m_componentLsb * (static_cast<double>(hyCompGct) + 0.5);

    // Convert to magnitude and angle
    double htMissMag = sqrt(hxCompGeV * hxCompGeV + hyCompGeV * hyCompGeV);
    double htMissAng = atan2(hyCompGeV, hxCompGeV);
    if (htMissAng < 0.0)
      htMissAng += 2.0 * M_PI;

    // Convert back to integer
    int htMissMagBits = static_cast<int>(m_etScale->rank(htMissMag)) & magnitudeMask;
    int htMissAngBits = static_cast<int>(htMissAng * 9.0 / M_PI) & angleMask;

    // Form the lut output
    result = (htMissMagBits << kHtMissAngleNBits) | htMissAngBits;
  }

  return result;
}

std::vector<double> L1GctHtMissLut::getThresholdsGeV() const { return m_etScale->getThresholds(); }

std::vector<unsigned> L1GctHtMissLut::getThresholdsGct() const {
  std::vector<unsigned> result;
  std::vector<double> thresholdsGeV = m_etScale->getThresholds();
  for (std::vector<double>::const_iterator thr = thresholdsGeV.begin(); thr != thresholdsGeV.end(); thr++) {
    result.push_back(static_cast<unsigned>((*thr) / (m_componentLsb)));
  }
  return result;
}

L1GctHtMissLut L1GctHtMissLut::operator=(const L1GctHtMissLut& lut) {
  const L1GctHtMissLut& temp(lut);
  return temp;
}

std::ostream& operator<<(std::ostream& os, const L1GctHtMissLut& lut) {
  os << "===L1GctHtMissLut===" << std::endl;
  std::vector<double> thresholds = lut.m_etScale->getThresholds();
  std::vector<double>::const_iterator thr = thresholds.begin();
  os << "Thresholds are: " << *(thr++);
  for (; thr != thresholds.end(); thr++) {
    os << ", " << *thr;
  }
  os << std::endl;
  os << "Max values for input to et scale " << lut.m_etScale->linScaleMax() << " and for output "
     << lut.m_etScale->rankScaleMax() << std::endl;
  os << "LSB used for conversion is " << lut.m_componentLsb << " GeV" << std::endl;
  os << "\n===Lookup table contents===\n" << std::endl;
  const L1GctLut<L1GctHtMissLut::NAddress, L1GctHtMissLut::NData>* temp = &lut;
  os << *temp;
  return os;
}

template class L1GctLut<L1GctHtMissLut::NAddress, L1GctHtMissLut::NData>;
