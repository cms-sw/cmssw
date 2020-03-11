#ifndef DQM_SiStripMonitorHardware_SiStripFEDSpyBuffer_H
#define DQM_SiStripMonitorHardware_SiStripFEDSpyBuffer_H

#include <string>
#include <ostream>
#include "EventFilter/SiStripRawToDigi/interface/SiStripFEDBufferComponents.h"
#include <cstdint>

namespace sistrip {

  //
  // Constants
  //

  static const uint16_t FEDCH_PER_DELAY_CHIP = 4;
  static const uint16_t DELAY_CHIPS_PER_FED = FEDCH_PER_FED / FEDCH_PER_DELAY_CHIP;
  static const uint16_t SPY_DELAY_CHIP_PAYLOAD_SIZE_IN_BYTES = 376 * 4;  // 376 32bit words
  static const uint16_t SPY_DELAY_CHIP_BUFFER_SIZE_IN_BYTES =
      SPY_DELAY_CHIP_PAYLOAD_SIZE_IN_BYTES + 8;                  // Extra 8 bytes for counters
  static const uint16_t SPY_DELAYCHIP_DATA_OFFSET_IN_BITS = 44;  // Offset to start of data
  //static const uint16_t SPY_SAMPLES_PER_CHANNEL = ( (SPY_DELAY_CHIP_BUFFER_SIZE_IN_BYTES * 8) - SPY_DELAYCHIP_DATA_OFFSET_IN_BITS ) / 10 / FEDCH_PER_DELAY_CHIP;
  // TW Dirty hack to lose the 3 samples from the end that screw things up...
  static const uint16_t SPY_SAMPLES_PER_CHANNEL = 298;
  static const uint16_t SPY_BUFFER_SIZE_IN_BYTES = SPY_DELAY_CHIP_BUFFER_SIZE_IN_BYTES * DELAY_CHIPS_PER_FED + 40;
  // Delaychip data + 8 bytes header for counters + 8 bytes for word with delay chip enable bits
  // + 16 bytes for DAQ header and trailer

  //
  // Class definitions
  //

  //class representing spy channel buffers
  class FEDSpyBuffer : public FEDBufferBase {
  public:
    /**
     * constructor from a FEDRawData buffer
     *
     * The sistrip::preconstructCheckFEDSpyBuffer() method should be used
     * to check the validity of fedBuffer before constructing a sistrip::FEDBuffer.
     *
     * @see sistrip::preconstructCheckFEDSpyBuffer()
     */
    //construct from buffer
    explicit FEDSpyBuffer(const FEDRawData& fedBuffer);
    ~FEDSpyBuffer() override;
    void print(std::ostream& os) const override;

    //get the run number from the corresponding global run
    uint32_t globalRunNumber() const;
    //get the L1 ID stored in the spy header
    uint32_t spyHeaderL1ID() const;
    //get the total frame count stored in the spy header
    uint32_t spyHeaderTotalEventCount() const;
    //get the L1 ID after reading a given delay chip
    uint32_t delayChipL1ID(const uint8_t delayChip) const;
    //get the total event count after reading a given delay chip
    uint32_t delayChipTotalEventCount(const uint8_t delayChip) const;

    //checks that a delay chip is complete i.e. that it all came from the same event
    bool delayChipGood(const uint8_t delayChip) const;
    //checks that a channel is usable (i.e. that the delay chip it is on is good)
    bool channelGood(const uint8_t internalFEDannelNum) const override;

  private:
    //mapping of channel index to position in data
    static const uint8_t channelPositionsInData_[FEDCH_PER_DELAY_CHIP];

    //setup the channel objects
    void findChannels();

    const uint8_t* payloadPointer_;
    uint16_t payloadLength_;
    uint8_t versionId_;
  };

  class FEDSpyChannelUnpacker {
  public:
    explicit FEDSpyChannelUnpacker(const FEDChannel& channel);
    uint16_t sampleNumber() const;
    uint16_t adc() const;
    bool hasData() const;
    FEDSpyChannelUnpacker& operator++();
    FEDSpyChannelUnpacker& operator++(int);

  private:
    const uint32_t* data_;
    size_t currentOffset_;
    uint16_t currentSample_;
    uint16_t valuesLeft_;
  };

  //
  // Inline function definitions
  //

  /**
   * Check if a FEDRawData object satisfies the requirements for constructing a sistrip::FEDSpyBuffer
   *
   * These are:
   *   - those from sistrip::preconstructCheckFEDBufferBase() (with checkRecognizedFormat equal to true)
   *   - the readout mode should be equal to sistrip::READOUT_MODE_SPY
   *
   * In case any check fails, a value different from sistrip::FEDBufferStatusCode::SUCCESS
   * is returned, and detailed information printed to LogDebug("FEDBuffer"), if relevant.
   *
   * @see sistrip::preconstructCheckFEDBufferBase()
   */
  inline FEDBufferStatusCode preconstructCheckFEDSpyBuffer(const FEDRawData& fedBuffer) {
    const auto st_base = preconstructCheckFEDBufferBase(fedBuffer, true);
    if (FEDBufferStatusCode::SUCCESS != st_base)
      return st_base;
    const TrackerSpecialHeader hdr{fedBuffer.data() + 8};
    if (READOUT_MODE_SPY != hdr.readoutMode())
      return FEDBufferStatusCode::EXPECT_SPY;
    return FEDBufferStatusCode::SUCCESS;
  }

  //FEDSpyChannelUnpacker

  inline FEDSpyChannelUnpacker::FEDSpyChannelUnpacker(const FEDChannel& channel)
      : data_(reinterpret_cast<const uint32_t*>(channel.data())),
        currentOffset_(channel.offset()),
        currentSample_(0),
        valuesLeft_(channel.length()) {}

  inline uint16_t FEDSpyChannelUnpacker::sampleNumber() const { return currentSample_; }

  inline bool FEDSpyChannelUnpacker::hasData() const { return (valuesLeft_ != 0); }

  inline FEDSpyChannelUnpacker& FEDSpyChannelUnpacker::operator++() {
    currentOffset_ += FEDCH_PER_DELAY_CHIP * 10;
    currentSample_++;
    valuesLeft_--;
    return (*this);
  }

  inline FEDSpyChannelUnpacker& FEDSpyChannelUnpacker::operator++(int) {
    ++(*this);
    return *this;
  }

}  // namespace sistrip

#endif  //ndef DQM_SiStripMonitorHardware_SiStripFEDSpyBuffer_H
