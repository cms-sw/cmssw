
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtTotal.h"

L1GctEtTotal::L1GctEtTotal() : m_data(0), m_bx(0) { }

L1GctEtTotal::L1GctEtTotal(uint16_t rawData) : m_data(rawData & kRawCtorMask), m_bx(0) { }

L1GctEtTotal::L1GctEtTotal(uint16_t rawData, int16_t bx) : m_data(rawData & kRawCtorMask), m_bx(bx) { }

L1GctEtTotal::L1GctEtTotal(unsigned et, bool oflow) : m_data(0),
                                                      m_bx(0)
{
  m_data = (et & kEtTotalMaxValue) | ((oflow || et>kEtTotalMaxValue) ? kEtTotalOFlowBit : 0);
}

L1GctEtTotal::L1GctEtTotal(unsigned et, bool oflow, int16_t bx) : m_data(0),
                                                                  m_bx(bx)
{
  m_data = (et & kEtTotalMaxValue) | ((oflow || et>kEtTotalMaxValue) ? kEtTotalOFlowBit : 0);
}

L1GctEtTotal::~L1GctEtTotal() { } 

/// Pretty-print operator for L1GctEtTotal
std::ostream& operator<<(std::ostream& s, const L1GctEtTotal& c) {
  s << " L1GctEtTotal: ";
  s << " et=" << c.et();
  if (c.overFlow()) { s << "; overflow set"; }
  return s;
}
