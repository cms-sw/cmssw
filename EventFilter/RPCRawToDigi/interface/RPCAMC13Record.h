#ifndef EventFilter_RPCRawToDigi_RPCAMC13Record_h
#define EventFilter_RPCRawToDigi_RPCAMC13Record_h

#include <cstdint>
#include <vector>

namespace rpcamc13 {

  class Header {  // http://joule.bu.edu/~hazen/CMS/AMC13/UpdatedDAQPath.pdf
  protected:
    static std::uint64_t const ufov_mask_ = 0xf000000000000000;
    static std::uint64_t const n_amc_mask_ = 0x00f0000000000000;
    static std::uint64_t const orbit_counter_mask_ = 0x0000000ffffffff0;

    static unsigned int const ufov_offset_ = 60;
    static unsigned int const n_amc_offset_ = 52;
    static unsigned int const orbit_counter_offset_ = 4;

  public:
    Header(std::uint64_t const record = 0x0);
    Header(unsigned int ufov, unsigned int n_amc, unsigned int orbit_counter);

    void set(std::uint64_t const record);
    void reset();

    std::uint64_t const& getRecord() const;

    unsigned int getFirmwareVersion() const;
    unsigned int getNAMC() const;
    unsigned int getOrbitCounter() const;

    void setFirmwareVersion(unsigned int ufov);
    void setNAMC(unsigned int n_amc);
    void setOrbitCounter(unsigned int orbit_counter);

  protected:
    std::uint64_t record_;
  };

  class Trailer {  // http://joule.bu.edu/~hazen/CMS/AMC13/UpdatedDAQPath.pdf
  protected:
    static std::uint64_t const crc_mask_ = 0xffffffff00000000;
    static std::uint64_t const block_number_mask_ = 0x000000000ff00000;
    static std::uint64_t const event_counter_mask_ = 0x00000000000ff000;
    static std::uint64_t const bx_counter_mask_ = 0x0000000000000fff;

    static unsigned int const crc_offset_ = 32;
    static unsigned int const block_number_offset_ = 20;
    static unsigned int const event_counter_offset_ = 12;
    static unsigned int const bx_counter_offset_ = 0;

  public:
    Trailer(std::uint64_t const record = 0x0);
    Trailer(std::uint32_t crc, unsigned int block_number, unsigned int event_counter, unsigned int bx_counter);

    void set(std::uint64_t const record);
    void reset();

    std::uint64_t const& getRecord() const;

    std::uint32_t getCRC() const;
    unsigned int getBlockNumber() const;
    unsigned int getEventCounter() const;
    unsigned int getBXCounter() const;

    void setCRC(std::uint32_t crc);
    void setBlockNumber(unsigned int block_number);
    void setEventCounter(unsigned int event_counter);
    void setBXCounter(unsigned int bx_counter);

  protected:
    std::uint64_t record_;
  };

  class AMCHeader {  // http://joule.bu.edu/~hazen/CMS/AMC13/UpdatedDAQPath.pdf
  protected:
    static std::uint64_t const length_incorrect_mask_ = 0x4000000000000000;
    static std::uint64_t const more_blocks_mask_ = 0x2000000000000000;  // is not last block
    static std::uint64_t const segmented_mask_ = 0x1000000000000000;    // is not first block
    static std::uint64_t const enabled_mask_ = 0x0800000000000000;
    static std::uint64_t const present_mask_ = 0x0400000000000000;
    static std::uint64_t const valid_mask_ = 0x0200000000000000;  // evn, bcn match
    static std::uint64_t const crc_ok_mask_ = 0x0100000000000000;
    static std::uint64_t const size_mask_ = 0x00ffffff00000000;
    static std::uint64_t const block_number_mask_ = 0x000000000ff00000;
    static std::uint64_t const amc_number_mask_ = 0x00000000000f0000;
    static std::uint64_t const board_id_mask_ = 0x000000000000ffff;

    static unsigned int const size_offset_ = 32;
    static unsigned int const block_number_offset_ = 20;
    static unsigned int const amc_number_offset_ = 16;
    static unsigned int const board_id_offset_ = 0;

    static unsigned int const size_limit_ = 0x1400;
    static unsigned int const size_max_ = 0x1000;

  public:
    AMCHeader(std::uint64_t const record = enabled_mask_ | present_mask_ | valid_mask_ | crc_ok_mask_);
    AMCHeader(bool length_correct,
              bool last_block,
              bool first_block,
              bool enabled,
              bool present,
              bool valid,
              bool crc_ok,
              unsigned int size,
              unsigned int block_number,
              unsigned int amc_number,
              unsigned int board_id);

    void set(std::uint64_t const record);
    void reset();

    std::uint64_t const& getRecord() const;

    bool isLengthCorrect() const;
    bool isLastBlock() const;
    bool isFirstBlock() const;
    bool isEnabled() const;
    bool isPresent() const;
    bool isValid() const;
    bool isCRCOk() const;
    unsigned int getSize() const;
    unsigned int getSizeInBlock() const;  // derived
    bool hasTotalSize() const;            // derived
    unsigned int getBlockNumber() const;
    unsigned int getAMCNumber() const;
    unsigned int getBoardId() const;

    void setLengthCorrect(bool length_correct);
    void setLastBlock(bool last_block);
    void setFirstBlock(bool first_block);
    void setEnabled(bool enabled);
    void setPresent(bool present);
    void setValid(bool valid);
    void setCRCOk(bool crc_ok);
    void setSize(unsigned int size);
    void setBlockNumber(unsigned int block_number);
    void setAMCNumber(unsigned int amc_number);
    void setBoardId(unsigned int board_id);

  protected:
    std::uint64_t record_;
  };

  class AMCPayload {
  public:
    AMCPayload();

    bool isValid() const;
    AMCHeader const& getAMCHeader() const;
    AMCHeader& getAMCHeader();
    std::vector<std::uint64_t> const& getData() const;
    std::vector<std::uint64_t>& getData();

    void setValid(bool valid);
    void setAMCHeader(AMCHeader const& header);
    void insert(std::uint64_t const* word_begin, unsigned int size);
    void clear();

  protected:
    bool valid_;
    AMCHeader amc_header_;
    std::vector<std::uint64_t> data_;
  };

}  // namespace rpcamc13

#include "EventFilter/RPCRawToDigi/interface/RPCAMC13Record.icc"

#endif  // EventFilter_RPCRawToDigi_RPCAMC13Record_h
