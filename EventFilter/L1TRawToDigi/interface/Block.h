#ifndef EventFilter_L1TRawToDigi_Block_h
#define EventFilter_L1TRawToDigi_Block_h

#include <memory>
#include <vector>

#include "EventFilter/L1TRawToDigi/interface/AMCSpec.h"
#include "DataFormats/L1Trigger/interface/BxBlock.h"

namespace l1t {
  enum block_t { MP7 = 0, CTP7, MTF7 };

  class BlockHeader {
  public:
    BlockHeader(unsigned int id, unsigned int size, unsigned int capID = 0, unsigned int flags = 0, block_t type = MP7)
        : id_(id), size_(size), capID_(capID), flags_(flags), type_(type){};
    // Create a MP7 block header: everything is contained in the raw uint32
    BlockHeader(const uint32_t* data)
        : id_((data[0] >> ID_shift) & ID_mask),
          size_((data[0] >> size_shift) & size_mask),
          capID_((data[0] >> capID_shift) & capID_mask),
          flags_((data[0] >> flags_shift) & flags_mask),
          type_(MP7){};

    bool operator<(const BlockHeader& o) const { return getID() < o.getID(); };

    unsigned int getID() const { return id_; };
    unsigned int getSize() const { return size_; };
    unsigned int getCapID() const { return capID_; };
    unsigned int getFlags() const { return flags_; };
    block_t getType() const { return type_; };

    uint32_t raw() const;

  private:
    static constexpr unsigned CTP7_shift = 0;
    static constexpr unsigned CTP7_mask = 0xffff;
    static constexpr unsigned ID_shift = 24;
    static constexpr unsigned ID_mask = 0xff;
    static constexpr unsigned size_shift = 16;
    static constexpr unsigned size_mask = 0xff;
    static constexpr unsigned capID_shift = 8;
    static constexpr unsigned capID_mask = 0xff;
    static constexpr unsigned flags_shift = 0;
    static constexpr unsigned flags_mask = 0xff;

    unsigned int id_;
    unsigned int size_;
    unsigned int capID_;
    unsigned int flags_;
    block_t type_;
  };

  class Block {
  public:
    Block(const BlockHeader& h, const uint32_t* payload_start, const uint32_t* payload_end)
        : header_(h), payload_(payload_start, payload_end){};
    Block(unsigned int id,
          const std::vector<uint32_t>& payload,
          unsigned int capID = 0,
          unsigned int flags = 0,
          block_t type = MP7)
        : header_(id, payload.size(), capID, flags, type), payload_(payload){};

    bool operator<(const Block& o) const { return header() < o.header(); };

    inline unsigned int getSize() const { return payload_.size() + 1; };

    BlockHeader header() const { return header_; };
    const std::vector<uint32_t>& payload() const { return payload_; };

    void amc(const amc::Header& h) { amc_ = h; };
    amc::Header amc() const { return amc_; };

    BxBlocks getBxBlocks(unsigned int payloadWordsPerBx, bool bxHeader) const;

  private:
    BlockHeader header_;
    amc::Header amc_;
    std::vector<uint32_t> payload_;
  };

  typedef std::vector<Block> Blocks;

  class Payload {
  public:
    Payload(const uint32_t* data, const uint32_t* end) : data_(data), end_(end), algo_(0), infra_(0){};
    virtual ~Payload(){};
    virtual unsigned getAlgorithmFWVersion() const { return algo_; };
    virtual unsigned getInfrastructureFWVersion() const { return infra_; };
    virtual unsigned getHeaderSize() const = 0;
    // Read header from data_ and advance data_ to point behind the
    // header.  Called by getBlock(), which also checks that data_ !=
    // end_ before calling (assumes size of one 32 bit word).
    virtual BlockHeader getHeader() = 0;
    virtual std::unique_ptr<Block> getBlock();

  protected:
    const uint32_t* data_;
    const uint32_t* end_;

    unsigned algo_;
    unsigned infra_;
  };

  class MP7Payload : public Payload {
  public:
    MP7Payload(const uint32_t* data, const uint32_t* end, bool legacy_mc = false);
    unsigned getHeaderSize() const override { return 1; };
    BlockHeader getHeader() override;
  };

  class MTF7Payload : public Payload {
  public:
    MTF7Payload(const uint32_t* data, const uint32_t* end);
    // Unused methods - we override getBlock() instead
    unsigned getHeaderSize() const override { return 0; };
    BlockHeader getHeader() override { return BlockHeader(nullptr); };
    std::unique_ptr<Block> getBlock() override;

  private:
    // sizes in 16 bit words
    static constexpr unsigned header_size = 12;
    static constexpr unsigned counter_size = 4;
    static constexpr unsigned trailer_size = 8;

    // maximum of the block length (64bits) and bit patterns of the
    // first bits (of 16bit words)
    static constexpr unsigned max_block_length_ = 3;
    static const std::vector<unsigned int> block_patterns_;

    int count(unsigned int pattern, unsigned int length) const;
    bool valid(unsigned int pattern) const;
  };

  class CTP7Payload : public Payload {
  public:
    CTP7Payload(const uint32_t* data, const uint32_t* end, amc::Header amcHeader);
    unsigned getHeaderSize() const override { return 2; };
    BlockHeader getHeader() override;
    std::unique_ptr<Block> getBlock() override;

  private:
    // FIXME check values
    static constexpr unsigned size_mask = 0xff;
    static constexpr unsigned size_shift = 16;

    unsigned size_;
    unsigned capId_;
    unsigned bx_per_l1a_;
    unsigned calo_bxid_;
    amc::Header amcHeader_;
  };
}  // namespace l1t

#endif
