#include "EventFilter/GctRawToDigi/src/GctBlockHeader.h"

GctBlockHeader::GctBlockHeader(
    uint32_t blockId, uint32_t blockLength, uint32_t nSamples, uint32_t bxId, uint32_t eventId, bool valid)
    : m_blockId(blockId),
      m_blockLength(blockLength),
      m_nSamples(nSamples),
      m_bxId(bxId),
      m_eventId(eventId),
      m_valid(valid) {}

std::ostream& operator<<(std::ostream& os, const GctBlockHeader& h) {
  os << "ID " << std::hex << h.blockId() << " : Length : " << h.blockLength() << " : Samples " << h.nSamples()
     << " : BX " << h.bxId() << " : Event " << h.eventId() << std::dec;
  return os;
}
