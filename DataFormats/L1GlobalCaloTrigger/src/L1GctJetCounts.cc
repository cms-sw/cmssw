
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
  m_data0(data0 & 0x7fff7fff), // Mask off bits 15 and 31 for better compression and consistency
  m_data1(data1 & 0x7fff7fff)  // with emulator constructor - these bits are not jet count data!
{
}

// constructor for emulator
L1GctJetCounts::L1GctJetCounts(vector<unsigned> counts) :
  m_data0(0), 
  m_data1(0)
{
  if (counts.size() != MAX_COUNTS) { }
  else {
    for (unsigned int i=0; i<3; ++i) {
      m_data0 += (counts[i]   << (5*i));
      m_data0 += (counts[i+3] << (5*i + 16));
      m_data1 += (counts[i+6] << (5*i));
      m_data1 += (counts[i+9] << (5*i + 16));
    }
  }
}

// destructor
L1GctJetCounts::~L1GctJetCounts()
{

}

// return counts by index
unsigned L1GctJetCounts::count(unsigned i) const
{
  if (i<6){ return ((m_data0 >> (i<3 ? (5*i) : ((5*i)+1))) & 0x1f); }
  else if (i < MAX_COUNTS) { return ((m_data1 >> (i<9 ? ((5*i)-30) : ((5*i)-29))) & 0x1f); }    
  else { return 0; }
}

// pretty print
ostream& operator<<(ostream& s, const L1GctJetCounts& c) {
  s << "L1GctJetCounts : " << endl;
  for (unsigned int i=0 ; i<12 ; ++i) {
    s << "     count " << i<< "=" << c.count(i) << endl;
  }
  return s;
}
