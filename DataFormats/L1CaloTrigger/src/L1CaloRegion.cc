
#include <boost/cstdint.hpp>
#include <ostream>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

using std::ostream;
using std::endl;

// default constructor
L1CaloRegion::L1CaloRegion() :
  m_data(0)
{
}

// constructor for emulation
L1CaloRegion::L1CaloRegion(unsigned et, bool overFlow, bool tauVeto, bool mip, bool quiet, unsigned crate, unsigned card, unsigned rgn)
{
  bool checkOvF = overFlow || (et>=0x400);
  m_data = 
    (et & 0x3ff) | 
    ((checkOvF) ? 0x400  : 0x0) |
    ((tauVeto)  ? 0x800  : 0x0) |
    ((mip)      ? 0x1000 : 0x0) |
    ((quiet)    ? 0x2000 : 0x0);
}

//constructor for unpacking
L1CaloRegion::L1CaloRegion(uint16_t data, unsigned crate, unsigned card, unsigned rgn) :
  m_data(data)
{

}

// destructor
L1CaloRegion::~L1CaloRegion() { }

// set mip bit
void L1CaloRegion::setMip(bool mip) {
  if (mip) { m_data |= 0x1000; }
  else { m_data &= 0xefff; }
}

// set quiet bit
void L1CaloRegion::setQuiet(bool quiet) {
  if (quiet) { m_data |= 0x2000; }
  else { m_data &= 0xdfff; }
}


// get RCT crate ID
unsigned L1CaloRegion::rctCrate() const { return 0; }

// get RCT reciever card ID (valid output for HB/HE)
unsigned L1CaloRegion::rctCard() const { return 0; }

// get RCT region index
unsigned L1CaloRegion::rctRegionIndex() const { return 0; }

// get local eta index (within RCT crate)
unsigned L1CaloRegion::rctEtaIndex() const { return 0; }

// get local phi index (within RCT crate)
unsigned L1CaloRegion::rctPhiIndex() const { return 0; } 

// get GCT source card ID
unsigned L1CaloRegion::gctCard() const { return 0; }

// get GCT eta index (global)
unsigned L1CaloRegion::gctEtaIndex() const { return 0; }

// get GCT phi index (global)
unsigned L1CaloRegion::gctPhiIndex() const { return 0; }

// get pseudorapidity
float L1CaloRegion::pseudorapidity() const { return 0.; }

// get phi in radians
float L1CaloRegion::phi() const { return 0.; }


// print to stream
ostream& operator << (ostream& os, const L1CaloRegion& reg) {
  os << "L1CaloRegion:";
  os << " Et=" << reg.et();
  os << " o/f=" << reg.overFlow();
  os << " tau=" << reg.tauVeto();
  os << " mip=" << reg.mip();
  os << " qt=" << reg.quiet();
  os << endl;
  os << "             ";
  os << " RCT crate=" << reg.rctCrate();
  os << " RCT crad=" << reg.rctCard();
  os << " RCT rgn=" << reg.rctRegionIndex();
  os << " RCT eta=" << reg.rctEtaIndex();
  os << " RCT phi=" << reg.rctPhiIndex();
  os << endl;
  os << "             ";
  os << " GCT card=" << reg.gctCard();
  os << " GCT eta=" << reg.gctEtaIndex();
  os << " GCT phi=" << reg.gctPhiIndex();
  os << endl;
  return os;
}

