#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternHtMiss.h"


// PUBLIC METHODS

// Default ctor
L1GctInternHtMiss::L1GctInternHtMiss():
  type_(nulltype),
  capBlock_(0),
  capIndex_(0),
  bx_(0),
  data_(0)
{
}

// Destructor
L1GctInternHtMiss::~L1GctInternHtMiss() {}

// Named ctor for making missing Ht x-component object from unpacker raw data.
L1GctInternHtMiss L1GctInternHtMiss::unpackerMissHtx(const uint16_t capBlock,
                                                     const uint16_t capIndex,
                                                     const int16_t bx,
                                                     const uint32_t data)
{
  return L1GctInternHtMiss(miss_htx, capBlock, capIndex, bx, data & kSingleComponentRawMask);
}

// Named ctor for making missing Ht y-component object from unpacker raw data.
L1GctInternHtMiss L1GctInternHtMiss::unpackerMissHty(const uint16_t capBlock,
                                                     const uint16_t capIndex,
                                                     const int16_t bx,
                                                     const uint32_t data)
{
  return L1GctInternHtMiss(miss_hty, capBlock, capIndex, bx, data & kSingleComponentRawMask);
}

// Named ctor for making missing Ht x & y components object from unpacker raw data.
L1GctInternHtMiss L1GctInternHtMiss::unpackerMissHtxHty(const uint16_t capBlock,
                                                        const uint16_t capIndex,
                                                        const int16_t bx,
                                                        const uint32_t data)
{
  return L1GctInternHtMiss(miss_htx_and_hty, capBlock, capIndex, bx, data & kDoubleComponentRawMask);
}

// Get Ht x-component
uint16_t L1GctInternHtMiss::htx() const
{
  if(type() == miss_htx) { return raw() & kSingleComponentHtMask; }
  if(type() == miss_htx_and_hty) { return raw() & kDoubleComponentHtMask; }
  return 0;
}

// Get Ht y-component
uint16_t L1GctInternHtMiss::hty() const
{
  if(type() == miss_hty) { return raw() & kSingleComponentHtMask; }
  if(type() == miss_htx_and_hty) { return (raw() >> kDoubleComponentHtyShift) & kDoubleComponentHtMask; }
  return 0;
}

// Get overflow
bool L1GctInternHtMiss::overflow() const
{
  if(type() == miss_htx || type() == miss_hty) { return (raw() & kSingleComponentOflowMask) != 0; }
  if(type() == miss_htx_and_hty) { return (raw() & kDoubleComponentOflowMask) != 0; }
  return false;
}


// PRIVATE METHODS

L1GctInternHtMiss::L1GctInternHtMiss(const L1GctInternHtMissType type,
                                     const uint16_t capBlock,
                                     const uint16_t capIndex,
                                     const int16_t bx,
                                     const uint32_t data):
  type_(type),
  capBlock_(capBlock),
  capIndex_(capIndex),
  bx_(bx),
  data_(data)
{
}


// PRETTY PRINTOUT OPERATOR

std::ostream& operator<<(std::ostream& os, const L1GctInternHtMiss& rhs)
{
  os << " L1GctInternHtMiss:  htx=";
  if(rhs.type() == L1GctInternHtMiss::miss_htx ||
     rhs.type() == L1GctInternHtMiss::miss_htx_and_hty)
  { os << "0x" << std::hex << rhs.htx() << std::dec; }
  else { os << "n/a"; }
  os << ", hty=";
  if(rhs.type() == L1GctInternHtMiss::miss_hty ||
     rhs.type() == L1GctInternHtMiss::miss_htx_and_hty)
  { os << "0x" << std::hex << rhs.hty() << std::dec; }
  else { os << "n/a"; }
  if (rhs.overflow()) { os << "; overflow set"; }
  return os;
}
