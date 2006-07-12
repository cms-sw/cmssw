
#include <boost/cstdint.hpp>
#include <ostream>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

using std::ostream;
using std::endl;

// default constructor
L1CaloRegion::L1CaloRegion() : m_id(), m_data(0) { }


// constructor for RCT emulator (HB/HE regions)
L1CaloRegion::L1CaloRegion(unsigned et, bool overFlow, bool tauVeto, bool mip, bool quiet, unsigned crate, unsigned card, unsigned rgn) :
  m_id(false, crate, card, rgn)
{
  pack(et, overFlow, tauVeto, mip, quiet);
}

// constructor for RCT emulator (HF regions)
L1CaloRegion::L1CaloRegion(unsigned et, bool fineGrain, unsigned crate, unsigned rgn) :
  m_id(false, crate, 0, rgn)
{
  pack((et & 0xff), (et >= 0xff), fineGrain, false, false);
}

// constructor from GCT card, region numbers
L1CaloRegion::L1CaloRegion(unsigned card, unsigned input, unsigned et, bool overFlow, bool fineGrain, bool mip, bool quiet) :
  m_id(false, card, input) // use constructor with dummy argument here (GCT card/input # NOT eta/phi!)
{
  pack(et, overFlow, fineGrain, mip, quiet);
}

// construct from global eta, phi indices
L1CaloRegion::L1CaloRegion(unsigned et, bool overFlow, bool fineGrain, bool mip, bool quiet, unsigned ieta, unsigned iphi) :
  m_id(ieta, iphi)
{
  pack(et, overFlow, fineGrain, mip, quiet);
}

//constructor for unpacking
L1CaloRegion::L1CaloRegion(uint16_t data, unsigned ieta, unsigned iphi) :
  m_id(ieta, iphi),
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

void L1CaloRegion::pack(unsigned et, bool overFlow, bool fineGrain, bool mip, bool quiet) {
  bool checkOvF = overFlow || (et>=0x400);
  m_data = 
    (et & 0x3ff) | 
    ((checkOvF)  ? 0x400  : 0x0) |
    ((fineGrain) ? 0x800  : 0x0) |
    ((mip)       ? 0x1000 : 0x0) |
    ((quiet)     ? 0x2000 : 0x0);
}

// print to stream
ostream& operator << (ostream& os, const L1CaloRegion& reg) {
  os << "L1CaloRegion:";
  os << " Et=" << reg.et();
  os << " o/f=" << reg.overFlow();
  os << " f/g=" << reg.fineGrain();
  os << " tau=" << reg.tauVeto();
  os << " mip=" << reg.mip();
  os << " qt=" << reg.quiet();
  os << endl;
  os << "             ";
  os << " RCT crate=" << reg.rctCrate();
  os << " RCT crad=" << reg.rctCard();
  os << " RCT rgn=" << reg.rctRegionIndex();
  os << " RCT eta=" << reg.rctEta();
  os << " RCT phi=" << reg.rctPhi();
  os << endl;
  os << "             ";
  os << " GCT card=" << reg.gctCard();
  os << " GCT eta=" << reg.gctEta();
  os << " GCT phi=" << reg.gctPhi();
  os << endl;
  os << "             ";
  os << " pseudorapidity=" << reg.pseudorapidity();
  os << " phi=" << reg.phi();
  os << endl;
  return os;
}

