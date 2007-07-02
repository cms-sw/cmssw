#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"

#include <iostream>

using std::ostream;
using std::endl;

// default constructor
L1CaloEmCand::L1CaloEmCand() :
  m_data(0),
  m_rctCrate(0),
  m_iso(false)
{ 

}

// construct from raw data (for use in unpacking)
 L1CaloEmCand::L1CaloEmCand(uint16_t data, unsigned crate, bool iso) :
   m_data(data),
   m_rctCrate(crate),
   m_iso(iso)
 {
 }

// construct from content (for use in emulator)
L1CaloEmCand::L1CaloEmCand(unsigned rank, unsigned region, unsigned card, unsigned crate, bool iso) : 
  m_rctCrate(crate),
  m_iso(iso) 
{
  m_data = (rank & 0x3f) + ((region & 0x1)<<6) + ((card & 0x7)<<7); 
}

// destructor
L1CaloEmCand::~L1CaloEmCand() { } 

// pretty print
ostream& operator<<(ostream& s, const L1CaloEmCand& cand) {
  s << "L1CaloEmCand : ";
  s << "rank=" << cand.rank();
  s << ", region=" << cand.rctRegion() << ", card=" << cand.rctCard() << ", crate=" << cand.rctCrate();
  s << ", iso=" << cand.isolated();
  return s;
}
