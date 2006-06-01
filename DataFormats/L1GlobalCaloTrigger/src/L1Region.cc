
#include <boost/cstdint.hpp>
#include <ostream>

#include "DataFormats/L1GlobalCaloTrigger/interface/L1Region.h"

using std::ostream;
using std::endl;

// default constructor
L1Region::L1Region() :
  m_eta(0),
  m_phi(0),
  m_rctCrate(0),
  m_data(0)
{
}

// constructor
L1Region::L1Region(unsigned eta, unsigned phi, unsigned et, bool overFlow, bool tauVeto, bool mip, bool quiet) :
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
L1Region::~L1Region() { }


// print to stream
ostream& operator << (ostream& os, const L1Region& reg) {
  os << "L1Region: eta=" << reg.eta();
  os << " phi=" << reg.phi();
  os << " Et=" << reg.et();
  os << " o/f=" << reg.overFlow();
  os << " tau=" << reg.tauVeto();
  os << " mip=" << reg.mip();
  os << " qt=" << reg.quiet();
  os << endl;
  return os;
}

