#ifndef EventFilter_L1TRawToDigi_Block_h
#define EventFilter_L1TRawToDigi_Block_h

#include <memory>
#include <vector>

#include "EventFilter/L1TRawToDigi/interface/AMCSpec.h"
#include "DataFormats/L1Trigger/interface/BxBlock.h"

namespace l1t {
  enum block_t { MP7 = 0, CTP7, MTF7 };
  namespace mtf7 {
    enum mtf7_block_t {
      // The "0b" prefix indicates binary; the block header id is stored in decimal.
      // Bits are the left-most bit (D15) of every 16-bit word in the format document.
      // Bottom-to-top in the document maps to left-to-right in each of the bit pattern.
      EvHd = 0b000111111111,  ///< Event Record Header   : block->header().getID() = 511
      CnBlk = 0b0010,         ///< Block of Counters     : block->header().getID() = 2
      ME = 0b0011,            ///< ME Data Record        : block->header().getID() = 3
      RPC = 0b0100,           ///< RPC Data Record       : block->header().getID() = 4
      GEM = 0b0111,           ///< GEM Data Record       : block->header().getID() = 7
      // FIXME, not currently defined... guess? JS - 01.07.20
      ME0 = 0b0110,        ///< ME0 Data Record       : block->header().getID() = 6
      SPOut = 0b01100101,  ///< SP Output Data Record : block->header().getID() = 101
      EvTr = 0b11111111    ///< Event Record Trailer  : block->header().getID() = 255
    };
  }

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

    /// Start of the EMTF DAQ payload, in number of 64-bit words
    static constexpr unsigned DAQ_PAYLOAD_OFFSET = 4;
    /// Maximum number of BX per MTF7 payload
    static constexpr unsigned MAX_BX_PER_PAYLOAD = 8;
    /// Maximum number of CSC words per MTF7 payload per bx: 9 links/sectors, 6 stations, 2 LCTs
    static constexpr unsigned ME_MAX_PER_BX = 108;
    /// Maximum number of RPC words per MTF7 payload per bx: 7 links/sectors, 6 stations, 2 segments
    static constexpr unsigned RPC_MAX_PER_BX = 84;
    /// Maximum number of GE1/1 words per MTF7 payload per bx: 7 GE1/1 links, 2 layers, 8 clusters
    static constexpr unsigned GE11_MAX_PER_BX = 112;
    /// TODO: Maximum number of GE2/1 words per MTF7 payload per bx: ?? GE2/1 links, 2 layers, ?? clusters
    static constexpr unsigned GE21_MAX_PER_BX = 0;
    /// TODO: Maximum number of ME0 words per MTF7 payload per bx: ?? ME0 links, ?? layers, ?? clusters
    static constexpr unsigned ME0_MAX_PER_BX = 0;
    /// Maximum number of SPz words per MTF7 payload per bx: 3 tracks, 2 words per track
    static constexpr unsigned SP_MAX_PER_BX = 6;
    /// Maximum number of 64-bit words in the EMTF payload
    static constexpr unsigned PAYLOAD_MAX_SIZE =
        MAX_BX_PER_PAYLOAD *
            (ME_MAX_PER_BX + RPC_MAX_PER_BX + GE11_MAX_PER_BX + GE21_MAX_PER_BX + ME0_MAX_PER_BX + SP_MAX_PER_BX) +
        (trailer_size / 4);

    static constexpr unsigned max_block_length_ = 3;         ///< maximum of the block length (64bits)
    static const std::vector<unsigned int> block_patterns_;  ///< bit patterns of the first bits (of 16bit words)

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
    unsigned six_hcal_feature_bits_;
    unsigned slot7_card_;
    amc::Header amcHeader_;
  };
}  // namespace l1t

#endif
