

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"


using std::ostream;
using std::string;
using std::hex;
using std::dec;

L1GctJetCand::L1GctJetCand() :
  m_data(0),
  m_isTau(false),
  m_isFor(false),
  m_captureBlock(0),
  m_captureIndex(0),
  m_bx(0)
{

}

//constructor for GT
L1GctJetCand::L1GctJetCand(uint16_t rawData, bool isTau, bool isFor) :
  m_data(rawData & 0x7fff), // 0x7fff is to mask off bit 15, which is not data that needs to be stored
  m_isTau(isTau),
  m_isFor(isFor),
  m_captureBlock(0),
  m_captureIndex(0),
  m_bx(0)
{
}

//constructor for GCT unpacker
L1GctJetCand::L1GctJetCand(uint16_t rawData, bool isTau, bool isFor, uint16_t block, uint16_t index, int16_t bx) : 
  m_data(rawData & 0x7fff), // 0x7fff is to mask off bit 15, which is not data that needs to be stored
  m_isTau(isTau),
  m_isFor(isFor),
  m_captureBlock(block&0xfff),
  m_captureIndex(index&0xff),
  m_bx(bx)
{
}

// constructor for use in emulator
// eta = -6 to -0, +0 to +6. Sign is bit 3, 1 means -ve Z, 0 means +ve Z
L1GctJetCand::L1GctJetCand(unsigned rank, unsigned phi, unsigned eta, bool isTau, bool isFor) : 
  m_data(0), // overridden below
  m_isTau(isTau),
  m_isFor(isFor),
  m_captureBlock(0),
  m_captureIndex(0),
  m_bx(0)
{ 
  m_data = (rank & 0x3f) + ((eta & 0xf)<<6) + ((phi & 0x1f)<<10); 
}

// constructor for use in emulator
// eta = -6 to -0, +0 to +6. Sign is bit 3, 1 means -ve Z, 0 means +ve Z
L1GctJetCand::L1GctJetCand(unsigned rank, unsigned phi, unsigned eta, bool isTau, bool isFor, uint16_t block, uint16_t index, int16_t bx) : 
  m_data(0), // overridden below
  m_isTau(isTau),
  m_isFor(isFor),
  m_captureBlock(block&0xfff),
  m_captureIndex(index&0xff),
  m_bx(bx)
{ 
  m_data = (rank & 0x3f) + ((eta & 0xf)<<6) + ((phi & 0x1f)<<10); 
}

L1GctJetCand::~L1GctJetCand() { } 

// return name
string L1GctJetCand::name() const { 
  if (m_isTau) { return "tau jet"; }
  else if (m_isFor) { return "forward jet"; }
  else { return "central jet"; }
}


// pretty print
ostream& operator<<(ostream& s, const L1GctJetCand& cand) {
  if (cand.empty()) {
    s << "L1GctJetCand empty jet";
  } else {
    s << "L1GctJetCand : ";
    s << "rank=" << cand.rank();
    s << ", etaSign=" << cand.etaSign() << ", eta=" << (cand.etaIndex()&0x7) << ", phi=" << cand.phiIndex();
    s << " type=";
    if (cand.isTau()) { s << "tau"; }
    else if (cand.isForward()) { s << "forward"; }
    else { s << "central"; }
  }
  s << hex << " cap block=" << cand.capBlock() << dec << ", index=" << cand.capIndex() << ", BX=" << cand.bx();
  return s;
}

L1CaloRegionDetId L1GctJetCand::regionId() const {

  // get global eta
  unsigned eta;
  if ( !isForward() ) {
    eta = ( etaSign()==1 ? 10-(etaIndex()&0x7)  : (etaIndex()&0x7)+11 );
  }
  else {
    eta = ( etaSign()==1 ? 3-(etaIndex()&0x7) : (etaIndex()&0x7)+18 );
  }

  return L1CaloRegionDetId(eta, phiIndex());

}


