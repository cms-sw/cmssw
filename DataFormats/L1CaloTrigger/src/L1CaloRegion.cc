
#include <boost/cstdint.hpp>
#include <ostream>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

using std::ostream;
using std::endl;

// constructor by ID
L1CaloRegion::L1CaloRegion(unsigned id, unsigned et, bool overFlow, bool tauVeto, bool mip, bool quiet) :
  m_id(id)
{
  bool checkOvF = overFlow || (et>=0x400);
  m_data = 
    (et & 0x3ff) | 
    ((checkOvF) ? 0x400  : 0x0) |
    ((tauVeto)  ? 0x800  : 0x0) |
    ((mip)      ? 0x1000 : 0x0) |
    ((quiet)    ? 0x2000 : 0x0);
}

// destructor
L1CaloRegion::~L1CaloRegion() { }

// return eta index
unsigned L1CaloRegion::etaIndex() const {
  return 0;
}

// return phi index
unsigned L1CaloRegion::phiIndex() const {
  return 0;
}

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

// print to stream
ostream& operator << (ostream& os, const L1CaloRegion& reg) {
  os << "L1CaloRegion: ";
  os << " iEta=" << reg.etaIndex();
  os << " iPhi=" << reg.phiIndex();
  os << " Et=" << reg.et();
  os << " o/f=" << reg.overFlow();
  os << " tau=" << reg.tauVeto();
  os << " mip=" << reg.mip();
  os << " qt=" << reg.quiet();
  os << endl;
  return os;
}

