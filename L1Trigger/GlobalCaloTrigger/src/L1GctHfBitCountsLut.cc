#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHfBitCountsLut.h"

//DEFINE STATICS
const int L1GctHfBitCountsLut::NAddress=L1GctHfLutSetup::kHfCountBits;
const int L1GctHfBitCountsLut::NData   =L1GctHfLutSetup::kHfOutputBits;

L1GctHfBitCountsLut::L1GctHfBitCountsLut(const L1GctHfLutSetup::hfLutType& type, const L1GctHfLutSetup* const fn) :
  L1GctLut<NAddress,NData>(),
  m_lutFunction(fn),
  m_lutType(type)
{
  if (fn != 0) m_setupOk = true;
}

L1GctHfBitCountsLut::L1GctHfBitCountsLut(const L1GctHfLutSetup::hfLutType& type) :
  L1GctLut<NAddress,NData>(),
  m_lutFunction(0),
  m_lutType(type)
{
}

L1GctHfBitCountsLut::L1GctHfBitCountsLut() :
  L1GctLut<NAddress,NData>(),
  m_lutFunction(0),
  m_lutType()
{
}

L1GctHfBitCountsLut::L1GctHfBitCountsLut(const L1GctHfBitCountsLut& lut) :
  L1GctLut<NAddress,NData>(),
  m_lutFunction(lut.lutFunction()),
  m_lutType(lut.lutType())
{
}

L1GctHfBitCountsLut::~L1GctHfBitCountsLut()
{
}

L1GctHfBitCountsLut L1GctHfBitCountsLut::operator= (const L1GctHfBitCountsLut& lut)
{
  L1GctHfBitCountsLut temp(lut);
  return temp;
}

std::ostream& operator << (std::ostream& os, const L1GctHfBitCountsLut& lut)
{
  os << "===L1GctHfBitCountsLut===" << std::endl;
  os << "\n===Lookup table contents===\n" << std::endl;
  const L1GctLut<L1GctHfBitCountsLut::NAddress,L1GctHfBitCountsLut::NData>* temp=&lut;
  os << *temp;
  return os;
}

template class L1GctLut<L1GctHfBitCountsLut::NAddress,L1GctHfBitCountsLut::NData>;


uint16_t L1GctHfBitCountsLut::value (const uint16_t lutAddress) const
{
  return m_lutFunction->outputValue(m_lutType, lutAddress) ;
}

