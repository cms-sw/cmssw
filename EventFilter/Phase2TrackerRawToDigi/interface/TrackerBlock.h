#ifndef EventFilter_Phase2TrackerRawToDigi_TrackerBlock_H
#define EventFilter_Phase2TrackerRawToDigi_TrackerBlock_H

#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2DAQFormatSpecification.h"
#include <vector>
#include <iostream>
#include <bitset>

using namespace Phase2DAQFormatSpecification;

// Base class to be then specialised for tracker header and trailer
class TrackerBlock {
public:
  explicit TrackerBlock(size_t nLines) : values_(nLines, 0) {}
  virtual ~TrackerBlock() = default;

  void setValue(std::vector<uint32_t> newValues) {
    assert(!newValues.empty() && "TrackerBlock: setValue() needs at least one word");
    values_ = std::move(newValues);
    is2S_ = extractBits(values_[0], N_BITS_PER_WORD - C_NUM_BITS_BOARD_TYPE, C_NUM_BITS_BOARD_TYPE) ==
            DTC_HEADER_OT_2S;
  }  

  bool is2S() const { return is2S_; }

  void printValues() const {
    for (size_t i = 0; i < values_.size(); ++i)
      printValue(i);
  }

  void printValue(size_t i) const {
    std::cout << blockName() << "[" << i << "]: " << values_[i] << "   " << std::bitset<32>(values_[i]) << std::endl;
  }

protected:
  std::vector<uint32_t> values_;
  bool is2S_{false};
  
  static uint32_t extractBits(uint32_t word, unsigned shift, unsigned width) {
    return (word >> shift) & ((1u << width) - 1u);
  }  

  virtual std::string blockName() const { return "TrackerBlock"; }

};

/**
 * @brief: Tracker Header Object, decodes Tracker Header described in the link below:
 * https://docs.google.com/spreadsheets/d/1RHZFqeHCoJhRaAfaKEO1Gx6U6c1Y3tRGhL_aSbZQROY/edit?gid=256168213#gid=256168213
 */
class TrackerHeader : public TrackerBlock {
public:
  TrackerHeader() : TrackerBlock(HEADER_N_LINES) {}

  explicit TrackerHeader(std::vector<uint32_t> words) : TrackerBlock(HEADER_N_LINES) { setValue(std::move(words)); }

  uint8_t  hasExtendedData() const { return static_cast<uint8_t>(getED()); }
  uint32_t getBoardType() const { return extractBits(values_[0], kShiftBoardType, C_NUM_BITS_BOARD_TYPE); }
  uint32_t getVersionMajor() const { return extractBits(values_[0], kShiftVerMajor, C_NUM_BITS_VERSION_MAJOR); }
  uint32_t getVersionMinor() const { return extractBits(values_[0], kShiftVerMinor, C_NUM_BITS_VERSION_MINOR); }
  uint32_t getMode() const { return extractBits(values_[0], kShiftMode, C_NUM_BITS_MODE); }
  uint32_t getED() const { return extractBits(values_[0], kShiftED, C_NUM_BITS_ED); }
  uint32_t getBoardID() const { return extractBits(values_[0], kShiftBoardID, C_NUM_BITS_BOARD_ID); }
  uint32_t getDAQpathCoreID() const { return extractBits(values_[0], kShiftCoreID, C_NUM_BITS_CORE_ID); }


  void printFields() const {
    printf(
        "Board Type: %02X, Ver Major: %03d, Ver Minor: %04d, Mode: %03d, ED: %01d, Board ID: %02X, DAQpath Core "
        "ID: %01X\n",
        getBoardType(),
        getVersionMajor(),
        getVersionMinor(),
        getMode(),
        getED(),
        getBoardID(),
        getDAQpathCoreID());
  }

protected:
  std::string blockName() const override { return "TrackerHeader"; }

private: 
  // pre-compute cumulative bit offsets according to TrackerHeader format
  // DAQpath CoreID | BoardID | ED | Mode | VerMinor | VerMajor | BoardType
  static constexpr int kShiftCoreID = 0;
  static constexpr int kShiftBoardID = kShiftCoreID + C_NUM_BITS_CORE_ID;
  static constexpr int kShiftED = kShiftBoardID + C_NUM_BITS_BOARD_ID;
  static constexpr int kShiftMode = kShiftED + C_NUM_BITS_ED;
  static constexpr int kShiftVerMinor = kShiftMode + C_NUM_BITS_MODE;
  static constexpr int kShiftVerMajor = kShiftVerMinor + C_NUM_BITS_VERSION_MINOR;
  static constexpr int kShiftBoardType = kShiftVerMajor + C_NUM_BITS_VERSION_MAJOR;

};



class TrackerTrailer : public TrackerBlock {
public:
  TrackerTrailer() : TrackerBlock(TRAILER_N_LINES) {}

  uint32_t getInvertedBoardType() const { return extractBits(values_[0], kShiftInvertedBoardType, C_NUM_BITS_BOARD_TYPE_INV); }

protected:
  std::string blockName() const override { return "TrackerTrailer"; }

private:
  // pre-compute cumulative bit offsets according to TrackerTrailer format
  // Reserved | BoardType
  static constexpr int kShiftInvertedBoardType = C_NUM_BITS_RESERVED_TRAILER;
  
};

#endif