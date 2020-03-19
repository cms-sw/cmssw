#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"

#include <iostream>

using std::dec;
using std::endl;
using std::hex;
using std::ostream;

// default constructor
L1CaloEmCand::L1CaloEmCand() : m_data(0), m_rctCrate(0), m_iso(false), m_index(0), m_bx(0) {}

// construct from raw data (for use in unpacking)
// last bool argument is a hack to distinguish this constructor from the next one!
L1CaloEmCand::L1CaloEmCand(uint16_t data, unsigned crate, bool iso)
    : m_data(data), m_rctCrate(crate), m_iso(iso), m_index(0), m_bx(0) {}

// construct from raw data (for use in unpacking)
// last bool argument is a hack to distinguish this constructor from the next one!
L1CaloEmCand::L1CaloEmCand(uint16_t data, unsigned crate, bool iso, uint16_t index, int16_t bx, bool dummy)
    : m_data(data), m_rctCrate(crate), m_iso(iso), m_index(index), m_bx(bx) {}

// construct from content (for use in emulator)
L1CaloEmCand::L1CaloEmCand(unsigned rank, unsigned region, unsigned card, unsigned crate, bool iso)
    : m_data(0),  // over-ridden below
      m_rctCrate(crate),
      m_iso(iso),
      m_index(0),
      m_bx(0)

{
  m_data = (rank & 0x3f) + ((region & 0x1) << 6) + ((card & 0x7) << 7);
}

// construct from content (for use in emulator)
L1CaloEmCand::L1CaloEmCand(
    unsigned rank, unsigned region, unsigned card, unsigned crate, bool iso, uint16_t index, int16_t bx)
    : m_data(0),  // over-ridden below
      m_rctCrate(crate),
      m_iso(iso),
      m_index(index),
      m_bx(bx) {
  m_data = (rank & 0x3f) + ((region & 0x1) << 6) + ((card & 0x7) << 7);
}

// destructor
L1CaloEmCand::~L1CaloEmCand() {}

void L1CaloEmCand::setBx(int16_t bx) { m_bx = bx; }

// pretty print
ostream& operator<<(ostream& s, const L1CaloEmCand& cand) {
  s << "L1CaloEmCand : ";
  s << "rank=" << cand.rank();
  s << ", region=" << cand.rctRegion() << ", card=" << cand.rctCard() << ", crate=" << cand.rctCrate();
  s << ", iso=" << cand.isolated();
  s << ", index=" << cand.index() << ", BX=" << cand.bx();
  return s;
}
