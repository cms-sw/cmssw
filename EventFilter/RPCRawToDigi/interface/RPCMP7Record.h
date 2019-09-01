#ifndef EventFilter_RPCRawToDigi_RPCMP7Record_h
#define EventFilter_RPCRawToDigi_RPCMP7Record_h

#include <cstdint>

#include "EventFilter/RPCRawToDigi/interface/RPCAMCRecord.h"

namespace rpcmp7 {

  class Header : public rpcamc::Header {
  public:
    static unsigned int const l1a_type_ = 0xd0;

  protected:
    // Second word
    static std::uint64_t const event_type_mask_ = 0x0000ffff00000000;
    static unsigned int const event_type_offset_ = 32;

  public:
    Header();
    Header(std::uint64_t const record[2]);
    Header(unsigned int amc_number,
           unsigned int event_counter,
           unsigned int bx_counter,
           unsigned int data_length,
           unsigned int orbit_counter,
           unsigned int board_id,
           unsigned int event_type = l1a_type_);
    ~Header() override;

    void reset() override;

    unsigned int getEventType() const;

    void setEventType(unsigned int event_type);
  };

  class SubHeader {
  protected:
    static std::uint64_t const algo_rev_mask_ = 0xffffffff00000000;
    static std::uint64_t const fw_rev_mask_ = 0x0000000000ffffff;

    static unsigned int const algo_rev_offset_ = 32;
    static unsigned int const fw_rev_offset_ = 0;

  public:
    SubHeader(std::uint64_t const record = 0x0);

    void set(std::uint64_t const word);
    void reset();

    std::uint64_t const& getRecord() const;

    unsigned int getAlgoVersion() const;
    unsigned int getFirmwareVersion() const;

    void setAlgoVersion(unsigned int algo_rev);
    void setFirmwareVersion(unsigned int fw_rev);

  protected:
    std::uint64_t record_;
  };

  class BlockHeader {
  protected:
    static std::uint32_t const id_mask_ = 0xff000000;
    static std::uint32_t const length_mask_ = 0x00ff0000;
    static std::uint32_t const caption_id_mask_ = 0x0000ff00;
    static std::uint32_t const zs_per_bx_mask_ = 0x00000002;
    static std::uint32_t const is_zs_mask_ = 0x00000001;

    static unsigned int const id_offset_ = 24;
    static unsigned int const length_offset_ = 16;
    static unsigned int const caption_id_offset_ = 8;
    static unsigned int const zs_per_bx_offset_ = 1;
    static unsigned int const is_zs_offset_ = 0;

  public:
    BlockHeader(std::uint32_t const record = 0x0);

    void set(std::uint32_t const record);
    void reset();

    std::uint32_t const& getRecord() const;

    unsigned int getId() const;
    unsigned int getLength() const;
    unsigned int getCaptionId() const;
    bool hasZeroSuppressionPerBX() const;
    bool isZeroSuppressed() const;

    bool isZeroSuppressionInverted() const;

    void setId(unsigned int id);
    void setLength(unsigned int length);
    void setCaptionId(unsigned int caption_id);
    void setZeroSuppressionPerBX(bool zs_per_bx);
    void setZeroSuppressed(bool is_zs);

    void setZeroSuppressionInverted(bool zs_inverted);

  protected:
    std::uint32_t record_;
  };

  class BXHeader {
  protected:
    static std::uint32_t const first_word_mask_ = 0xff000000;
    static std::uint32_t const total_length_mask_ = 0x00ff0000;
    static std::uint32_t const is_zs_mask_ = 0x00000001;

    static unsigned int const first_word_offset_ = 24;
    static unsigned int const total_length_offset_ = 16;
    static unsigned int const is_zs_offset_ = 0;

  public:
    BXHeader(std::uint32_t const record = 0x0);

    void set(std::uint32_t const record);
    void reset();

    std::uint32_t const& getRecord() const;

    unsigned int getFirstWord() const;
    unsigned int getTotalLength() const;
    bool isZeroSuppressed() const;

    // https://twiki.cern.ch/twiki/pub/CMS/MP7ZeroSuppression/mp7_zs_payload_formats.pdf
    unsigned int getBXId() const;
    unsigned int getTotalBX() const;

    void setFirstWord(unsigned int first_word);
    void setTotalLength(unsigned int length);
    void setZeroSuppressed(bool zs);

  protected:
    std::uint32_t record_;
  };

}  // namespace rpcmp7

#include "EventFilter/RPCRawToDigi/interface/RPCMP7Record.icc"

#endif  // EventFilter_RPCRawToDigi_RPCMP7Record_h
