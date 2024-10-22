#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternEmCand.h"

#include <iostream>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

using std::dec;
using std::hex;
using std::ostream;
using std::string;

// default constructor
L1GctInternEmCand::L1GctInternEmCand() : m_data(0), m_captureBlock(0), m_captureIndex(0), m_iso(false), m_bx(0) {}

// construct from raw data (for use in unpacking)
L1GctInternEmCand::L1GctInternEmCand(uint16_t data, bool iso, unsigned block, unsigned index, int16_t bx)
    : m_data(data), m_captureBlock(block & 0xfff), m_captureIndex(index & 0xff), m_iso(iso), m_bx(bx) {}

// construct from eta/phi etc
L1GctInternEmCand::L1GctInternEmCand(
    unsigned rank, unsigned eta, unsigned etaSgn, unsigned phi, bool iso, unsigned block, unsigned index, int16_t bx)
    : m_data(0),  // Over-ridden in construct()
      m_captureBlock(block & 0xfff),
      m_captureIndex(index & 0xff),
      m_iso(iso),
      m_bx(bx) {
  construct(rank, eta, etaSgn, phi);
}

// destructor
L1GctInternEmCand::~L1GctInternEmCand() {}

// name of candidate type
string L1GctInternEmCand::name() const { return (isolated() ? "iso EM" : "non iso EM"); }

// was a candidate found
bool L1GctInternEmCand::empty() const { return (rank() == 0); }

// pretty print
ostream& operator<<(ostream& s, const L1GctInternEmCand& cand) {
  s << "L1GctInternEmCand : ";
  s << "rank=" << cand.rank();
  s << ", etaSign=" << cand.etaSign() << ", eta=" << (cand.etaIndex() & 0xf) << ", phi=" << cand.phiIndex();
  s << ", iso=" << cand.isolated();
  s << " cap block=" << hex << cand.capBlock();
  s << ", index=" << dec << cand.capIndex() << ", BX=" << cand.bx();
  return s;
}

// return region object
L1CaloRegionDetId L1GctInternEmCand::regionId() const {
  // get global eta
  unsigned eta = (etaSign() == 1 ? 10 - (etaIndex() & 0xf) : 11 + (etaIndex() & 0xf));
  return L1CaloRegionDetId(eta, phiIndex());
}

// construct from rank, eta, phi
void L1GctInternEmCand::construct(unsigned rank, unsigned eta, unsigned etaSgn, unsigned phi) {
  m_data = (rank & 0x3f) + ((eta & 0xf) << 6) + ((etaSgn & 0x1) << 10) + ((phi & 0x1f) << 11);
}
