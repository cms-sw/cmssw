
#include <boost/cstdint.hpp>
#include <ostream>

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctRegion.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctMap.h"

using std::ostream;
using std::endl;

// constructor by ID
L1GctRegion::L1GctRegion(unsigned id, unsigned et, bool overFlow, bool tauVeto, bool mip, bool quiet) :
  m_id(id)
{
  m_data = 
    (et & 0x3ff) | 
    ((overFlow) ? 0x400  : 0x0) |
    ((tauVeto)  ? 0x800  : 0x0) |
    ((mip)      ? 0x1000 : 0x0) |
    ((quiet)    ? 0x2000 : 0x0);
}

// destructor
L1GctRegion::~L1GctRegion() { }

// get eta index
unsigned L1GctRegion::eta() const { 
  return L1GctMap::getMap()->eta(*this); 
}

// get phi index
unsigned L1GctRegion::phi() const { 
  return L1GctMap::getMap()->phi(*this); 
}

// get RCT crate ID
unsigned L1GctRegion::rctCrate() const { 
  return L1GctMap::getMap()->rctCrate(*this); 
}

// get source card ID
unsigned L1GctRegion::sourceCard() const { 
  return L1GctMap::getMap()->sourceCard(*this); 
}

// set mip bit
void L1GctRegion::setMip(bool mip) {
  if (mip) { m_data |= 0x1000; }
  else { m_data &= 0xefff; }
}

// set quiet bit
void L1GctRegion::setQuiet(bool quiet) {
  if (quiet) { m_data |= 0x2000; }
  else { m_data &= 0xdfff; }
}

// print to stream
ostream& operator << (ostream& os, const L1GctRegion& reg) {
  os << "L1GctRegion: eta=" << reg.eta();
  os << " phi=" << reg.phi();
  os << " Et=" << reg.et();
  os << " o/f=" << reg.overFlow();
  os << " tau=" << reg.tauVeto();
  os << " mip=" << reg.mip();
  os << " qt=" << reg.quiet();
  os << endl;
  return os;
}

