#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"

#include <iostream>

using std::ostream;
using std::string;

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
// eta = -6 to -0, +0 to +6. Sign is bit 3, 1 means -ve Z, 0 means +ve Z
L1GctEmCand::L1GctEmCand(unsigned rank, unsigned eta, unsigned phi, bool iso) : 
  m_iso(iso) 
{
  m_data = (rank & 0x3f) + ((eta & 0xf)<<6) + ((phi & 0x1f)<<10); 
}

// destructor
L1GctEmCand::~L1GctEmCand() { } 

// name of candidate type
string L1GctEmCand::name() const {
  return (isolated() ? "iso EM" : "non iso EM" ); 
}

// was a candidate found
bool L1GctEmCand::empty() const { 
  return (rank() == 0); 
}

// pretty print
ostream& operator<<(ostream& s, const L1GctEmCand& cand) {
  s << "L1GctEmCand : ";
  s << "rank=" << cand.rank();
  s << ", eta=" << cand.etaIndex() << ", phi=" << cand.phiIndex();
  s << ", iso=" << cand.isolated();
  return s;
}
