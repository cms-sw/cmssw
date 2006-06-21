#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"

#include <iostream>

using std::ostream;
using std::endl;

// default constructor
L1CaloEmCand::L1CaloEmCand() :
  m_data(0),
  m_iso(false)
{ 

}

// construct from raw data (for use in unpacking)
L1CaloEmCand::L1CaloEmCand(uint16_t data) :
  m_data(data),
  m_iso(false)
{

}

// construct from content (for use in emulator)
L1CaloEmCand::L1CaloEmCand(int rank, int phi, int eta, bool iso) : 
  m_iso(iso) 
{
  m_data = (rank & 0x3f) + ((phi & 0x1f)<<6) + ((eta & 0xf)<<11); 
}

// destructor
L1CaloEmCand::~L1CaloEmCand() { } 

// pretty print
ostream& operator<<(ostream& s, const L1CaloEmCand& cand) {
  s << "L1CaloRegion : ";
  s << "rank=" << cand.rank() << ", eta=" << cand.eta() << ", phi=" << cand.phi();
  s << ", iso=" << cand.isolated();
  return s;
}
