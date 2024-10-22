#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHfBitCountsLut.h"

//DEFINE STATICS
const int L1GctHfBitCountsLut::NAddress = 5;
const int L1GctHfBitCountsLut::NData = 3;

L1GctHfBitCountsLut::L1GctHfBitCountsLut(const L1GctHfEtSumsLut::hfLutType& type)
    : L1GctLut<NAddress, NData>(), m_lutType(type) {
  // No setup required
  m_setupOk = true;
}

L1GctHfBitCountsLut::L1GctHfBitCountsLut() : L1GctLut<NAddress, NData>(), m_lutType() {
  // No setup required
  m_setupOk = true;
}

L1GctHfBitCountsLut::L1GctHfBitCountsLut(const L1GctHfBitCountsLut& lut)
    : L1GctLut<NAddress, NData>(), m_lutType(lut.lutType()) {
  // No setup required
  m_setupOk = true;
}

L1GctHfBitCountsLut::~L1GctHfBitCountsLut() {}

uint16_t L1GctHfBitCountsLut::value(const uint16_t lutAddress) const {
  // Return "address=data" up to the maximum number of output codes
  const int maxOutput = ((1 << NData) - 1);
  if (lutAddress > maxOutput)
    return maxOutput;
  else
    return (lutAddress & maxOutput);
}

std::vector<unsigned> L1GctHfBitCountsLut::getThresholdsGct() const {
  std::vector<unsigned> result;
  // Return "address=data" up to the maximum number of output codes
  for (unsigned add = 1; add < (1 << NData); add++) {
    result.push_back(add);
  }
  return result;
}

L1GctHfBitCountsLut L1GctHfBitCountsLut::operator=(const L1GctHfBitCountsLut& lut) {
  const L1GctHfBitCountsLut& temp(lut);
  return temp;
}

std::ostream& operator<<(std::ostream& os, const L1GctHfBitCountsLut& lut) {
  os << "===L1GctHfBitCountsLut===" << std::endl;
  os << "\n===Lookup table contents===\n" << std::endl;
  const L1GctLut<L1GctHfBitCountsLut::NAddress, L1GctHfBitCountsLut::NData>* temp = &lut;
  os << *temp;
  return os;
}

template class L1GctLut<L1GctHfBitCountsLut::NAddress, L1GctHfBitCountsLut::NData>;
