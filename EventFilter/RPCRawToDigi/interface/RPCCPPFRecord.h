#ifndef EventFilter_RPCRawToDigi_RPCCPPFRecord_h
#define EventFilter_RPCRawToDigi_RPCCPPFRecord_h

#include <cstdint>

namespace rpccppf {

  class RXRecord {
  protected:
    static std::uint32_t const link_mask_ = 0xfe000000;
    static std::uint32_t const bx_counter_mod_mask_ = 0x01f00000;
    static std::uint32_t const da_mask_ = 0x00080000;
    static std::uint32_t const de_mask_ = 0x00040000;
    static std::uint32_t const eod_mask_ = 0x00020000;
    static std::uint32_t const delay_mask_ = 0x0001c000;
    static std::uint32_t const link_board_mask_ = 0x00003000;
    static std::uint32_t const connector_mask_ = 0x00000e00;
    static std::uint32_t const partition_mask_ = 0x00000100;
    static std::uint32_t const partition_data_mask_ = 0x000000ff;

    static unsigned int const link_offset_ = 25;
    static unsigned int const bx_counter_mod_offset_ = 20;
    static unsigned int const da_offset_ = 19;
    static unsigned int const de_offset_ = 18;
    static unsigned int const eod_offset_ = 17;
    static unsigned int const delay_offset_ = 14;
    static unsigned int const link_board_offset_ = 12;
    static unsigned int const connector_offset_ = 9;
    static unsigned int const partition_offset_ = 8;
    static unsigned int const partition_data_offset_ = 0;

  public:
    RXRecord(std::uint32_t const record = da_mask_);

    void set(std::uint32_t const record);
    void reset();

    std::uint32_t const& getRecord() const;

    unsigned int getLink() const;
    unsigned int getBXCounterMod() const;
    bool isAcknowledge() const;
    bool isError() const;
    bool isEOD() const;
    unsigned int getDelay() const;
    unsigned int getLinkBoard() const;
    unsigned int getConnector() const;
    unsigned int getPartition() const;
    std::uint8_t getPartitionData() const;

    void setLink(unsigned int link);
    void setBXCounterMod(unsigned int bx);
    void setAcknowledge(bool da);
    void setError(bool de);
    void setEOD(bool eod);
    void setDelay(unsigned int delay);
    void setLinkBoard(unsigned int link_board);
    void setConnector(unsigned int connector);
    void setPartition(unsigned int partition);
    void setPartitionData(std::uint8_t data);

  protected:
    std::uint32_t record_;
  };

  class TXRecord {
  protected:
    static std::uint32_t const theta_mask_[2];
    static std::uint32_t const phi_mask_[2];

    static unsigned int const theta_offset_[2];
    static unsigned int const phi_offset_[2];

  public:
    TXRecord();
    TXRecord(std::uint32_t const record);

    void set(std::uint32_t const record);
    void reset();

    std::uint32_t const& getRecord() const;

    unsigned int getTheta(unsigned int index) const;
    unsigned int getPhi(unsigned int index) const;
    bool isValid(unsigned int index) const;  // derived

    void setTheta(unsigned int index, unsigned int theta);
    void setPhi(unsigned int index, unsigned int phi);

  protected:
    std::uint32_t record_;
  };

}  // namespace rpccppf

#include "EventFilter/RPCRawToDigi/interface/RPCCPPFRecord.icc"

#endif  // EventFilter_RPCRawToDigi_RPCCPPFRecord_h
