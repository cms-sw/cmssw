#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternEtSum.h"
#include <cstdint>

L1GctInternEtSum::L1GctInternEtSum() {}

/// construct from individual quantities
L1GctInternEtSum::L1GctInternEtSum(uint16_t capBlock, uint16_t capIndex, int16_t bx, uint32_t et, uint8_t oflow)
    : type_(null), capBlock_(capBlock), capIndex_(capIndex), bx_(bx), data_(0) {
  this->setEt(et);
  this->setOflow(oflow);
}

/// destructor
L1GctInternEtSum::~L1GctInternEtSum() {}

L1GctInternEtSum L1GctInternEtSum::fromJetTotEt(const uint16_t capBlock,
                                                const uint16_t capIndex,
                                                const int16_t bx,
                                                const uint32_t data) {
  L1GctInternEtSum s;
  s.setEt(data & kTotEtOrHtMaxValue);
  s.setOflow((data >> kTotEtOrHtNBits) & 0x1);
  s.setCapBlock(capBlock);
  s.setCapIndex(capIndex);
  s.setBx(bx);
  s.setType(jet_tot_et);
  return s;
}

L1GctInternEtSum L1GctInternEtSum::fromJetTotHt(const uint16_t capBlock,
                                                const uint16_t capIndex,
                                                const int16_t bx,
                                                const uint32_t data) {
  L1GctInternEtSum s;
  uint32_t word = data >> 16;
  s.setEt(word & kTotEtOrHtMaxValue);
  s.setOflow((word >> kTotEtOrHtNBits) & 0x1);
  s.setCapBlock(capBlock);
  s.setCapIndex(capIndex);
  s.setBx(bx);
  s.setType(jet_tot_ht);
  return s;
}

L1GctInternEtSum L1GctInternEtSum::fromJetMissEt(const uint16_t capBlock,
                                                 const uint16_t capIndex,
                                                 const int16_t bx,
                                                 const uint32_t data) {
  L1GctInternEtSum s;
  s.setEt(data & kJetMissEtMaxValue);
  s.setOflow((data >> kJetMissEtNBits) & 0x1);
  s.setCapBlock(capBlock);
  s.setCapIndex(capIndex);
  s.setBx(bx);
  s.setType(jet_miss_et);
  return s;
}

L1GctInternEtSum L1GctInternEtSum::fromTotalEtOrHt(const uint16_t capBlock,
                                                   const uint16_t capIndex,
                                                   const int16_t bx,
                                                   const uint32_t data) {
  L1GctInternEtSum s;
  s.setEt(data & kTotEtOrHtMaxValue);
  s.setOflow((data >> kTotEtOrHtNBits) & 0x1);
  s.setCapBlock(capBlock);
  s.setCapIndex(capIndex);
  s.setBx(bx);
  s.setType(total_et_or_ht);
  return s;
}

L1GctInternEtSum L1GctInternEtSum::fromMissEtxOrEty(const uint16_t capBlock,
                                                    const uint16_t capIndex,
                                                    const int16_t bx,
                                                    const uint32_t data) {
  L1GctInternEtSum s;
  s.setEt(data & kMissExOrEyNBits);
  s.setOflow(0);  // No over flow bit at the moment
  s.setCapBlock(capBlock);
  s.setCapIndex(capIndex);
  s.setBx(bx);
  s.setType(miss_etx_or_ety);
  return s;
}

/// Emulator constructors

L1GctInternEtSum L1GctInternEtSum::fromEmulatorJetTotEt(unsigned totEt, bool overFlow, int16_t bx) {
  L1GctInternEtSum s;
  s.setEt(totEt & kTotEtOrHtMaxValue);
  if (overFlow || (totEt > kTotEtOrHtMaxValue))
    s.setOflow(0x1);
  s.setBx(bx);
  s.setType(jet_tot_et);
  return s;
}

L1GctInternEtSum L1GctInternEtSum::fromEmulatorJetTotHt(unsigned totHt, bool overFlow, int16_t bx) {
  L1GctInternEtSum s;
  s.setEt(totHt & kTotEtOrHtMaxValue);
  if (overFlow || (totHt > kTotEtOrHtMaxValue))
    s.setOflow(0x1);
  s.setBx(bx);
  s.setType(jet_tot_ht);
  return s;
}

L1GctInternEtSum L1GctInternEtSum::fromEmulatorJetMissEt(int missEtxOrEty, bool overFlow, int16_t bx) {
  L1GctInternEtSum s;
  s.setEt(missEtxOrEty & kJetMissEtMaxValue);
  if (overFlow || (missEtxOrEty >= kJetMissEtOFlowBit / 2) || (missEtxOrEty < -kJetMissEtOFlowBit / 2))
    s.setOflow(0x1);
  s.setBx(bx);
  s.setType(jet_miss_et);
  return s;
}

L1GctInternEtSum L1GctInternEtSum::fromEmulatorTotalEtOrHt(unsigned totEtOrHt, bool overFlow, int16_t bx) {
  L1GctInternEtSum s;
  s.setEt(totEtOrHt & kTotEtOrHtMaxValue);
  if (overFlow || (totEtOrHt > kTotEtOrHtMaxValue))
    s.setOflow(0x1);
  s.setBx(bx);
  s.setType(total_et_or_ht);
  return s;
}

L1GctInternEtSum L1GctInternEtSum::fromEmulatorMissEtxOrEty(int missEtxOrEty, bool overFlow, int16_t bx) {
  L1GctInternEtSum s;
  s.setEt(missEtxOrEty & kMissExOrEyMaxValue);
  if (overFlow || (missEtxOrEty >= kMissExOrEyOFlowBit / 2) || (missEtxOrEty < -kMissExOrEyOFlowBit / 2))
    s.setOflow(0x1);
  s.setBx(bx);
  s.setType(miss_etx_or_ety);
  return s;
}

/// equality operator
bool L1GctInternEtSum::operator==(const L1GctInternEtSum& c) const { return (data_ == c.raw() && bx_ == c.bx()); }

/// set value
void L1GctInternEtSum::setValue(uint32_t val) {
  data_ &= 0x80000000;
  data_ |= val & 0x7ffffff;
}

/// set et
void L1GctInternEtSum::setEt(uint32_t et) { setValue(et); }

/// set count
void L1GctInternEtSum::setCount(uint32_t count) { setValue(count); }

/// set overflow bit
void L1GctInternEtSum::setOflow(uint8_t oflow) {
  data_ &= 0x7ffffff;
  data_ |= (oflow & 0x1) << 31;
}

/// Pretty-print operator for L1GctInternEtSum
std::ostream& operator<<(std::ostream& s, const L1GctInternEtSum& c) {
  s << "L1GctInternEtSum : ";

  if (c.type() == L1GctInternEtSum::jet_miss_et) {
    s << " type=jet_miss_et";
  } else if (c.type() == L1GctInternEtSum::jet_tot_et) {
    s << " type=jet_tot_et";
  } else if (c.type() == L1GctInternEtSum::jet_tot_ht) {
    s << " type=jet_tot_ht";
  } else if (c.type() == L1GctInternEtSum::total_et_or_ht) {
    s << " type=total_et_or_ht";
  } else if (c.type() == L1GctInternEtSum::miss_etx_or_ety) {
    s << " type=miss_etx_or_ety";
  }

  if (c.empty()) {
    s << " empty!";
  } else {
    s << " mag=" << c.et();
    if (c.oflow()) {
      s << " overflow set";
    }
  }

  s << " cap block=" << std::hex << c.capBlock();
  s << " index=" << std::dec << c.capIndex();
  s << " BX=" << c.bx();

  return s;
}
