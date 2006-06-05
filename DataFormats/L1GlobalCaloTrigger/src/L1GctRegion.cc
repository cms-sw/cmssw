
#include <boost/cstdint.hpp>
#include <ostream>

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctRegion.h"

using std::ostream;
using std::endl;

// constructor
L1GctRegion::L1GctRegion(unsigned eta, unsigned phi, unsigned et, bool overFlow, bool tauVeto, bool mip, bool quiet) :
  m_eta(eta),
  m_phi(phi)
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


// set mip bit
L1GctRegion::setMip(bool mip) {
  if (mip) { m_data |= 0x1000; }
  else { m_data &= 0xefff; }
}

// set quiet bit
L1GctRegion::setQuiet(bool quiet) {
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

