

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

using std::ostream;
using std::endl;
using std::hex;
using std::dec;

// default constructor
L1CaloRegion::L1CaloRegion() : m_id(), m_data(0), m_bx(0) { }


// constructor for RCT emulator (HB/HE regions)
L1CaloRegion::L1CaloRegion(unsigned et, bool overFlow, bool tauVeto, bool mip, bool quiet, unsigned crate, unsigned card, unsigned rgn) :
  m_id(crate, card, rgn),
  m_data(0), // over-ridden below
  m_bx(0)
{
  pack(et, overFlow, tauVeto, mip, quiet);
}

// constructor for RCT emulator (HF regions)
L1CaloRegion::L1CaloRegion(unsigned et, bool fineGrain, unsigned crate, unsigned rgn) :
  m_id(crate, 999, rgn),
  m_data(0), // over-ridden below
  m_bx(0)
{
  pack((et & 0xff), (et >= 0xff), fineGrain, false, false);
}

// construct from global eta, phi indices
L1CaloRegion::L1CaloRegion(unsigned et,
			   bool overFlow, 
			   bool fineGrain, 
			   bool mip, 
			   bool quiet, 
			   unsigned ieta, 
			   unsigned iphi) :
  m_id(ieta, iphi),
  m_data(0), // over-ridden below
  m_bx(0)
{
  pack(et, overFlow, fineGrain, mip, quiet);
}

//constructor for unpacking
L1CaloRegion::L1CaloRegion(uint16_t data, unsigned ieta, unsigned iphi, int16_t bx) :
  m_id(ieta, iphi),
  m_data(data),
  m_bx(bx)
{

}

// destructor
L1CaloRegion::~L1CaloRegion() { }

// named ctors

// for HB/HE from RCT indices
L1CaloRegion L1CaloRegion::makeHBHERegion(unsigned et, 
					  bool overFlow, 
					  bool tauVeto, 
					  bool mip, 
					  bool quiet, 
					  unsigned crate, 
					  unsigned card, 
					  unsigned rgn) {
  L1CaloRegion r;
  r.setRegionId( L1CaloRegionDetId(crate, card, rgn) );
  r.setBx(0);
  r.pack(et, overFlow, tauVeto, mip, quiet);
  return r;
}

// for HF from RCT indices
L1CaloRegion L1CaloRegion::makeHFRegion(unsigned et, 
					 bool fineGrain, 
					 unsigned crate, 
					 unsigned rgn) {
  L1CaloRegion r;
  r.setRegionId( L1CaloRegionDetId(crate, 999, rgn) );
  r.setBx(0);
  r.pack((et & 0xff), (et >= 0xff), fineGrain, false, false);
  return r;
}

// HB/HE/HF from GCT indices
L1CaloRegion L1CaloRegion::makeRegionFromGctIndices(unsigned et, 
						    bool overFlow, 
						    bool fineGrain, 
						    bool mip, 
						    bool quiet, 
						    unsigned ieta, 
						    unsigned iphi) {
  L1CaloRegion r;
  r.setRegionId( L1CaloRegionDetId(ieta, iphi) );
  r.setBx(0);
  r.pack(et, overFlow, fineGrain, mip, quiet);
  return r;
}

//constructor for unpacking
L1CaloRegion L1CaloRegion::makeRegionFromUnpacker(uint16_t data, 
                                                  unsigned ieta, 
                                                  unsigned iphi, 
                                                  uint16_t block, 
                                                  uint16_t index, 
                                                  int16_t bx)
{
  L1CaloRegion r;
  r.setRegionId( L1CaloRegionDetId(ieta,iphi) );
  r.setRawData(data);
  r.setCaptureBlock(block);
  r.setCaptureIndex(index);
  r.setBx(bx);
  return r;
}

L1CaloRegion L1CaloRegion::makeGctJetRegion(const unsigned et, 
                                            const bool overFlow, 
                                            const bool fineGrain,
                                            const unsigned ieta, 
                                            const unsigned iphi,
                                            const int16_t bx) {
  L1CaloRegion r;
  r.setRegionId( L1CaloRegionDetId(ieta, iphi) );
  r.setBx(bx);
  r.pack12BitsEt(et, overFlow, fineGrain, false, false);
  return r;

}

// set BX
void L1CaloRegion::setBx(int16_t bx) {
  m_bx = bx;
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

void L1CaloRegion::pack(unsigned et, bool overFlow, bool fineGrain, bool mip, bool quiet) {
  bool checkOvF = overFlow || (et>=0x400);
  m_data = 
    (et & 0x3ff) | 
    ((checkOvF)  ? 0x400  : 0x0) |
    ((fineGrain) ? 0x800  : 0x0) |
    ((mip)       ? 0x1000 : 0x0) |
    ((quiet)     ? 0x2000 : 0x0);
}

void L1CaloRegion::pack12BitsEt(unsigned et, bool overFlow, bool fineGrain, bool mip, bool quiet) {
  bool checkOvF = overFlow || (et>=0x400);
  m_data = 
    (et & 0xfff) | 
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
  os << " tau=" << reg.tauVeto() << endl;
  os << " RCT crate=" << reg.rctCrate();
  os << " RCT card=" << reg.rctCard();
  os << " RCT rgn=" << reg.rctRegionIndex();
  os << " RCT eta=" << reg.rctEta();
  os << " RCT phi=" << reg.rctPhi() << endl;
  os << " GCT eta=" << reg.gctEta();
  os << " GCT phi=" << reg.gctPhi() << endl;
  os << hex << " cap block=" << reg.capBlock() << dec << ", index=" << reg.capIndex() << ", BX=" << reg.bx();
  return os;
}

