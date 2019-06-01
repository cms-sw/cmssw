#ifndef EventFilter_RPCRawToDigi_RPCAMCRecord_h
#define EventFilter_RPCRawToDigi_RPCAMCRecord_h

#include <cstdint>

namespace rpcamc {

  class Header {  // http://joule.bu.edu/~hazen/CMS/AMC13/UpdatedDAQPath.pdf
  protected:
    // First Word
    static std::uint64_t const amc_number_mask_ = 0x0f00000000000000;
    static std::uint64_t const event_counter_mask_ = 0x00ffffff00000000;
    static std::uint64_t const bx_counter_mask_ = 0x00000000fff00000;
    static std::uint64_t const data_length_mask_ = 0x00000000000fffff;

    static unsigned int const amc_number_offset_ = 56;
    static unsigned int const event_counter_offset_ = 32;
    static unsigned int const bx_counter_offset_ = 20;
    static unsigned int const data_length_offset_ = 0;

    // Second word
    static std::uint64_t const orbit_counter_mask_ = 0x00000000ffff0000;
    static std::uint64_t const board_id_mask_ = 0x000000000000ffff;

    static unsigned int const orbit_counter_offset_ = 16;
    static unsigned int const board_id_offset_ = 0;

  public:
    Header();
    Header(std::uint64_t const record[2]);
    Header(unsigned int amc_number,
           unsigned int event_counter,
           unsigned int bx_counter,
           unsigned int data_length,
           unsigned int orbit_counter,
           unsigned int board_id);
    virtual ~Header();

    void set(unsigned int nword, std::uint64_t const word);
    virtual void reset();

    std::uint64_t const* getRecord() const;

    unsigned int getAMCNumber() const;
    unsigned int getEventCounter() const;
    unsigned int getBXCounter() const;
    unsigned int getDataLength() const;
    bool hasDataLength() const;  // derived

    unsigned int getOrbitCounter() const;
    unsigned int getBoardId() const;

    void setAMCNumber(unsigned int amc_number);
    void setEventCounter(unsigned int event_counter);
    void setBXCounter(unsigned int bx_counter);
    void setDataLength(unsigned int data_length);

    void setOrbitCounter(unsigned int orbit_counter);
    void setBoardId(unsigned int board_id);

  protected:
    std::uint64_t record_[2];
  };

  class Trailer {  // http://joule.bu.edu/~hazen/CMS/AMC13/UpdatedDAQPath.pdf
  protected:
    static std::uint64_t const crc_mask_ = 0xffffffff00000000;
    static std::uint64_t const event_counter_mask_ = 0x00000000ff000000;
    static std::uint64_t const data_length_mask_ = 0x00000000000fffff;

    static unsigned int const crc_offset_ = 32;
    static unsigned int const event_counter_offset_ = 24;
    static unsigned int const data_length_offset_ = 0;

  public:
    Trailer(std::uint64_t const record = 0x0);
    Trailer(std::uint32_t crc, unsigned int event_counter, unsigned int data_length);

    void set(std::uint64_t const record);
    void reset();

    std::uint64_t const& getRecord() const;

    std::uint32_t getCRC() const;
    unsigned int getEventCounter() const;
    unsigned int getDataLength() const;

    void setCRC(std::uint32_t crc);
    void setEventCounter(unsigned int event_counter);
    void setDataLength(unsigned int data_length);

  protected:
    std::uint64_t record_;
  };

}  // namespace rpcamc

#include "EventFilter/RPCRawToDigi/interface/RPCAMCRecord.icc"

#endif  // EventFilter_RPCRawToDigi_RPCAMCRecord_h
