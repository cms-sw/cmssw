
#include <boost/cstdint.hpp>
#include <ostream>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

using std::ostream;
using std::endl;

// default constructor
L1CaloRegion::L1CaloRegion() : m_id(), m_data(0) { }


// constructor for emulation : HB/HE regions
L1CaloRegion::L1CaloRegion(unsigned et, bool overFlow, bool tauVeto, bool mip, bool quiet, unsigned crate, unsigned card, unsigned rgn) :
  m_id(crate, card, rgn)
{
  bool checkOvF = overFlow || (et>=0x400);
  m_data = 
    (et & 0x3ff) | 
    ((checkOvF) ? 0x400  : 0x0) |
    ((tauVeto)  ? 0x800  : 0x0) |
    ((mip)      ? 0x1000 : 0x0) |
    ((quiet)    ? 0x2000 : 0x0);
}

// constructor for emulation : HF regions
L1CaloRegion::L1CaloRegion(unsigned et, bool overFlow, bool fineGrain, unsigned crate, unsigned rgn) :
  m_id()
{
  bool checkOvF = overFlow || (et>=0x400);
  m_data = 
    (et & 0x3ff) | 
    ((checkOvF) ? 0x400  : 0x0) |
    ((fineGrain)  ? 0x800  : 0x0);

  // calculate eta, phi
  // int ieta=0;
  // int iphi=0;
  // id=L1CaloRegionDetId(ieta, iphi);

}


// construct from global eta, phi indices
L1CaloRegion::L1CaloRegion(unsigned et, bool overFlow, bool tauVeto, bool mip, bool quiet, int eta, int phi) :
  m_id(eta, phi)
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
L1CaloRegion::L1CaloRegion(uint16_t data, int eta, int phi) :
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

