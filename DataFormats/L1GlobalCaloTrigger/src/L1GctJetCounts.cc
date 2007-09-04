
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCounts.h"

using std::vector;
using std::ostream;
using std::endl;

// default constructor
L1GctJetCounts::L1GctJetCounts() :
  m_data0(0),
  m_data1(0)
{

}

// constructor for unpacking
L1GctJetCounts::L1GctJetCounts(uint32_t data0, uint32_t data1) :
  m_data0(data0),
  m_data1(data1)
{

}

// constructor for emulator
L1GctJetCounts::L1GctJetCounts(vector<unsigned> counts) :
  m_data0(0),
  m_data1(0)
{
  if (counts.size() != 12) {
  }
  else {
    for (int i=0; i<6; i++) {
      m_data0 += counts[i]<<(i*5);
      m_data1 += counts[i+6]<<(i*5);
    }
  }
}

// destructor
L1GctJetCounts::~L1GctJetCounts()
{

}

// return counts by index
unsigned L1GctJetCounts::count(unsigned i) const {
  if (i>=0 && i<6) {
    return (m_data0>>(i*5)) & 0x1f;
  }
  else if (i>=6 && i<12) {
    return (m_data1>>((i-6)*5)) & 0x1f;
  }    
  else {
    return 0;
  }
}

// pretty print
ostream& operator<<(ostream& s, const L1GctJetCounts& c) {
  s << "L1GctJetCounts : " << endl;
  for (int i=0; i<12; i++) {
    s << "     count " << i<< "=" << c.count(i) << endl;
  }
  return s;
}
