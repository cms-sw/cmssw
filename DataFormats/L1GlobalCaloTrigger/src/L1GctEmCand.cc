#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"

#include <iostream>

using std::ostream;

// default constructor
L1GctEmCand::L1GctEmCand() :
  m_data(0),
  m_iso(false)
{ 

}

// construct from raw data (for use in unpacking)
 L1GctEmCand::L1GctEmCand(uint16_t data, bool iso) :
   m_data(data),
   m_iso(iso)
 {

 }

// construct from content (for use in emulator)
L1GctEmCand::L1GctEmCand(unsigned rank, int eta, int phi, bool iso) : 
  m_iso(iso) 
{
  // need to include sign bit!
  m_data = (rank & 0x3f) + ((eta & 0x7)<<6) + ((static_cast<unsigned>(phi) & 0xf)<<10); 
}

// destructor
L1GctEmCand::~L1GctEmCand() { } 

// pretty print
ostream& operator<<(ostream& s, const L1GctEmCand& cand) {
  s << "L1GctEmCand : ";
  s << "rank=" << cand.rank();
  s << ", eta=" << cand.level1EtaIndex() << ", phi=" << cand.level1PhiIndex();
  s << ", iso=" << cand.isolated();
  return s;
}
