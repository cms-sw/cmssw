#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"

#include <iostream>


using std::ostream;
using std::string;
using std::hex;
using std::dec;

// default constructor
L1GctEmCand::L1GctEmCand() :
  m_data(0),
  m_iso(false),
  m_captureBlock(0),
  m_captureIndex(0),
  m_bx(0)
{ 

}

// construct from raw data, no source (i.e. no capBlock/capIndex); used in GT
L1GctEmCand::L1GctEmCand(uint16_t rawData, bool iso) :
  m_data(rawData & 0x7fff), // 0x7fff is to mask off bit 15, which is not data that needs to be stored
  m_iso(iso),
  m_captureBlock(0),
  m_captureIndex(0),
  m_bx(0)
{

}

// construct from raw data with source - used in GCT unpacker
 L1GctEmCand::L1GctEmCand(uint16_t rawData, bool iso, uint16_t block, uint16_t index, int16_t bx) :
   m_data(rawData & 0x7fff), // 0x7fff is to mask off bit 15, which is not data that needs to be stored
   m_iso(iso),
   m_captureBlock(block&0xfff),
   m_captureIndex(index&0xff),
   m_bx(bx)
{

}

// construct from content - used in GCT emulator
// eta = -6 to -0, +0 to +6. Sign is bit 3, 1 means -ve Z, 0 means +ve Z
L1GctEmCand::L1GctEmCand(unsigned rank, unsigned phi, unsigned eta, bool iso) : 
  m_data(0), // override below
  m_iso(iso),
  m_captureBlock(0),
  m_captureIndex(0),
  m_bx(0)
 
{
  construct(rank, eta, phi);
}

// construct from content, with source (i.e. capBlock/capIndex); will be used in GCT emulator one day?
// eta = -6 to -0, +0 to +6. Sign is bit 3, 1 means -ve Z, 0 means +ve Z
L1GctEmCand::L1GctEmCand(unsigned rank, unsigned phi, unsigned eta, bool iso, uint16_t block, uint16_t index, int16_t bx) : 
  m_data(0), // override below
  m_iso(iso),
  m_captureBlock(block&0xfff),
  m_captureIndex(index&0xff),
  m_bx(bx)
{
  construct(rank, eta, phi);
}

// construct from RCT output candidate
L1GctEmCand::L1GctEmCand(L1CaloEmCand& c) :
  m_data(0), // override below
  m_iso(c.isolated()),
  m_captureBlock(0),
  m_captureIndex(0),
  m_bx(c.bx())
{
  unsigned eta=((c.regionId().rctEta() & 0x7) | (c.regionId().ieta()<11 ? 0x8 : 0x0));
  construct(c.rank(), eta, c.regionId().iphi());
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

// return region object
L1CaloRegionDetId L1GctEmCand::regionId() const {
  // get global eta
  unsigned eta = ( etaSign()==1 ? 10-(etaIndex()&0x7) : 11+(etaIndex()&0x7) );
  return L1CaloRegionDetId(eta, phiIndex());
}

// construct from rank, eta, phi
void L1GctEmCand::construct(unsigned rank, unsigned eta, unsigned phi) {
  if (rank>0) {
    m_data = (rank & 0x3f) + ((eta & 0xf)<<6) + ((phi & 0x1f)<<10);
  } else {
    // Default values for zero rank electrons,
    // different in hardware for positive and negative eta
    if ((eta & 0x8)==0) { m_data = 0x7000; } else { m_data = 0x7400; }
  }
}

// pretty print
ostream& operator<<(ostream& s, const L1GctEmCand& cand) {
  s << "L1GctEmCand : ";
  s << "rank=" << cand.rank();
  s << ", etaSign=" << cand.etaSign() << ", eta=" << (cand.etaIndex()&0x7) << ", phi=" << cand.phiIndex();
  s << ", iso=" << cand.isolated();
  s << hex << " cap block=" << cand.capBlock() << dec << ", index=" << cand.capIndex() << ", BX=" << cand.bx();
  return s;
}

unsigned L1GctEmCand::phiIndex() const { return (m_data>>10) & 0x1f; } 
unsigned L1GctEmCand::etaIndex() const { return (m_data>>6) & 0xf; } 
unsigned L1GctEmCand::rank() const { return m_data & 0x3f; }


