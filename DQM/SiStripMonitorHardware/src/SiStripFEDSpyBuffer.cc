#include "DQM/SiStripMonitorHardware/interface/SiStripFEDSpyBuffer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace sistrip {

  const uint8_t FEDSpyBuffer::channelPositionsInData_[FEDCH_PER_DELAY_CHIP] = {0, 3, 2, 1};

  FEDSpyBuffer::FEDSpyBuffer(const FEDRawData& fedBuffer)
      : FEDBufferBase(fedBuffer),
        payloadPointer_(getPointerToDataAfterTrackerSpecialHeader() + 16),
        payloadLength_(getPointerToByteAfterEndOfPayload() - payloadPointer_),
        versionId_(*(getPointerToDataAfterTrackerSpecialHeader() + 3)) {
    //Check the buffer format version ID and take action for any exceptions
    if (versionId_ == 0x00) {
      payloadPointer_ = payloadPointer_ - 8;
    }
    //find the channel start positions
    findChannels();
  }

  FEDSpyBuffer::~FEDSpyBuffer() {}

  void FEDSpyBuffer::findChannels() {
    size_t delayChipStartByteIndex = 0;
    //Loop over delay chips checking their data fits into buffer and setting up channel objects with correct offset
    for (uint8_t iDelayChip = 0; iDelayChip < DELAY_CHIPS_PER_FED; ++iDelayChip) {
      if (delayChipStartByteIndex + SPY_DELAY_CHIP_BUFFER_SIZE_IN_BYTES > payloadLength_) {
        throw cms::Exception("FEDSpyBuffer") << "Delay chip " << uint16_t(iDelayChip) << " does not fit into buffer. "
                                             << "Buffer size is " << bufferSize() << " delay chip data starts at "
                                             << delayChipStartByteIndex + 8 + 8 + 8 + 8 << ". ";
      }
      for (uint8_t i = 0; i < FEDCH_PER_DELAY_CHIP; i++) {
        const uint8_t chanelIndexInDataOrder = channelPositionsInData_[i];
        const uint8_t fedCh = iDelayChip * FEDCH_PER_DELAY_CHIP + i;
        const size_t channelOffsetInBits = SPY_DELAYCHIP_DATA_OFFSET_IN_BITS + 10 * chanelIndexInDataOrder;
        channels_[fedCh] =
            FEDChannel(payloadPointer_ + delayChipStartByteIndex, channelOffsetInBits, SPY_SAMPLES_PER_CHANNEL);
      }
      delayChipStartByteIndex += SPY_DELAY_CHIP_BUFFER_SIZE_IN_BYTES;
    }
  }

  uint32_t FEDSpyBuffer::globalRunNumber() const {
    if (versionId_ < 0x02) {
      return 0;
    }
    const uint8_t* runNumberPointer = getPointerToDataAfterTrackerSpecialHeader() + 4;
    uint32_t result = 0;
    result |= runNumberPointer[0];
    result |= (uint32_t(runNumberPointer[1]) << 8);
    result |= (uint32_t(runNumberPointer[2]) << 16);
    result |= (uint32_t(runNumberPointer[3]) << 24);
    return result;
  }

  uint32_t FEDSpyBuffer::spyHeaderL1ID() const {
    if (versionId_ == 0x00) {
      return delayChipL1ID(0);
    }
    uint32_t result = 0;
    const uint8_t* spyCounters = payloadPointer_ - 8;
    result |= spyCounters[4];
    result |= (uint32_t(spyCounters[5]) << 8);
    result |= (uint32_t(spyCounters[6]) << 16);
    result |= (uint32_t(spyCounters[7]) << 24);
    return result;
  }

  uint32_t FEDSpyBuffer::spyHeaderTotalEventCount() const {
    if (versionId_ == 0x00) {
      return delayChipTotalEventCount(0);
    }
    uint32_t result = 0;
    const uint8_t* spyCounters = payloadPointer_ - 8;
    result |= spyCounters[0];
    result |= (uint32_t(spyCounters[1]) << 8);
    result |= (uint32_t(spyCounters[2]) << 16);
    result |= (uint32_t(spyCounters[3]) << 24);
    return result;
  }

  uint32_t FEDSpyBuffer::delayChipL1ID(const uint8_t delayChip) const {
    const uint8_t* delayChipCounters = payloadPointer_ + ((SPY_DELAY_CHIP_BUFFER_SIZE_IN_BYTES) * (delayChip + 1) - 8);
    uint32_t result = 0;
    result |= delayChipCounters[4];
    result |= (uint32_t(delayChipCounters[5]) << 8);
    result |= (uint32_t(delayChipCounters[6]) << 16);
    result |= (uint32_t(delayChipCounters[7]) << 24);
    return result;
  }

  uint32_t FEDSpyBuffer::delayChipTotalEventCount(const uint8_t delayChip) const {
    const uint8_t* delayChipCounters = payloadPointer_ + ((SPY_DELAY_CHIP_BUFFER_SIZE_IN_BYTES) * (delayChip + 1) - 8);
    uint32_t result = 0;
    result |= delayChipCounters[0];
    result |= (uint32_t(delayChipCounters[1]) << 8);
    result |= (uint32_t(delayChipCounters[2]) << 16);
    result |= (uint32_t(delayChipCounters[3]) << 24);
    return result;
  }

  void FEDSpyBuffer::print(std::ostream& os) const {
    FEDBufferBase::print(os);
    //TODO
  }

  bool FEDSpyBuffer::delayChipGood(const uint8_t delayChip) const {
    if (versionId_ == 0x00) {
      if (delayChip == 0)
        return true;
    }
    uint32_t l1CountBefore = 0;
    uint32_t totalEventCountBefore = 0;
    if (delayChip == 0) {
      l1CountBefore = spyHeaderL1ID();
      totalEventCountBefore = spyHeaderTotalEventCount();
    } else {
      l1CountBefore = delayChipL1ID(delayChip - 1);
      totalEventCountBefore = delayChipTotalEventCount(delayChip - 1);
    }
    const uint32_t l1CountAfter = delayChipL1ID(delayChip);
    const uint32_t totalEventCountAfter = delayChipTotalEventCount(delayChip);
    const bool eventMatches = ((l1CountBefore == l1CountAfter) && (totalEventCountBefore == totalEventCountAfter));
    if (!eventMatches) {
      std::ostringstream ss;
      ss << "Delay chip data was overwritten on chip " << uint16_t(delayChip) << " L1A before: " << l1CountBefore
         << " after: " << l1CountAfter << " Total event count before: " << totalEventCountBefore
         << " after: " << totalEventCountAfter << std::endl;
      dump(ss);
      edm::LogInfo("FEDSpyBuffer") << ss.str();
    }
    return eventMatches;
  }

  bool FEDSpyBuffer::channelGood(const uint8_t internalFEDChannelNum) const {
    return delayChipGood(internalFEDChannelNum / FEDCH_PER_DELAY_CHIP);
  }

  uint16_t FEDSpyChannelUnpacker::adc() const {
    const size_t offsetWords = currentOffset_ / 32;
    const uint8_t offsetBits = currentOffset_ % 32;
    if (offsetBits < 23) {
      return ((data_[offsetWords] >> (32 - 10 - offsetBits)) & 0x3FF);
    } else {
      return (((data_[offsetWords] << (10 - 32 + offsetBits)) & 0x3FF) |
              ((data_[offsetWords + 1] & (0xFFC00000 << (32 - offsetBits))) >> (64 - 10 - offsetBits)));
    }
  }

}  // namespace sistrip
