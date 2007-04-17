#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"

#include <iostream>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

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
  construct(rank, eta, phi);
}

// construct from RCT output candidate
L1GctEmCand::L1GctEmCand(L1CaloEmCand& c) :
  m_iso(c.isolated())
{
  construct(c.rank(), c.regionId().gctEta(), c.regionId().gctPhi());
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

// return region object
L1CaloRegionDetId L1GctEmCand::regionId() const {
  // get global eta
  unsigned eta = ( etaSign()==1 ? 10-(etaIndex()&0x7) : 11+(etaIndex()&0x7) );
  return L1CaloRegionDetId(eta, phiIndex());
}

// construct from rank, eta, phi
void L1GctEmCand::construct(unsigned rank, unsigned eta, unsigned phi) {
  m_data = (rank & 0x3f) + ((eta & 0xf)<<6) + ((phi & 0x1f)<<10);
}
