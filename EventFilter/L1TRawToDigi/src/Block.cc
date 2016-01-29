#include <iomanip>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/L1TRawToDigi/interface/Block.h"

#define EDM_ML_DEBUG 1

namespace l1t {
  
  const std::vector<unsigned int> 
  MTF7Payload::block_patterns_ = 
    {
      // The "0b" prefix indicates binary; the block header id is stored in decimal.
      // Bits are the left-most bit (D15) of every 16-bit word in the format document.
      // Bottom-to-top in the document maps to left-to-right in each of the block_patterns_
      0b000111111111, // Event Record Header   : block->header().getID() = 511
      // Left-most bits of 0xA and 0x9 are both 1 in binary
      0b0010,         // Block of Counters     : block->header().getID() = 2
      0b0011,         // ME Data Record        : block->header().getID() = 3
      0b0100,         // RPC Data Record       : block->header().getID() = 4
      0b01100101,     // SP Output Data Record : block->header().getID() = 101
      0b11111111      // Event Record Trailer  : block->header().getID() = 255
      // Left-most bits of 0xF and 0xE are both 1 in binary
    };
  
   uint32_t
   BlockHeader::raw(block_t type) const
   {
      if (type_ == MP7) {
         LogTrace("L1T") << "Writing MP7 link header";
         return ((id_ & ID_mask) << ID_shift) | ((size_ & size_mask) << size_shift) | ((capID_ & capID_mask) << capID_shift);
      }
      // if (type_ == MTF7) {
      //    LogTrace("L1T") << "Writing MTF7 link header";
      //    return ((id_ & ID_mask) << ID_shift) | ((size_ & size_mask) << size_shift) | ((capID_ & capID_mask) << capID_shift);
      // }
      LogTrace("L1T") << "Writing CTP7 link header";
      return ((id_ & CTP7_mask) << CTP7_shift);
   }

   std::auto_ptr<Block>
   Payload::getBlock()
   {
      if (end_ - data_ < getHeaderSize()) {
         LogDebug("L1T") << "Reached end of payload";
         return std::auto_ptr<Block>();
      }

      if (data_[0] == 0xffffffff) {
         LogDebug("L1T") << "Skipping padding word";
         ++data_;
         return getBlock();
      }

      auto header = getHeader();

      if (end_ - data_ < header.getSize()) {
         edm::LogError("L1T")
            << "Expecting a block size of " << header.getSize()
            << " but only " << (end_ - data_) << " words remaining";
         return std::auto_ptr<Block>();
      }

      LogTrace("L1T") << "Creating block with size " << header.getSize();

      auto res = std::auto_ptr<Block>(new Block(header, data_, data_ + header.getSize()));
      data_ += header.getSize();
      return res;
   }

   MP7Payload::MP7Payload(const uint32_t * data, const uint32_t * end, bool legacy_mc) : Payload(data, end)
   {
      // For legacy MC (74 first MC campaigns) skip one empty word that was
      // reserved for the header.  With data, read out infrastructure
      // version and algorithm version.
      if (legacy_mc) {
         LogTrace("L1T") << "Skipping " << std::hex << *data_;
         ++data_;
      } else {
         infra_ = data_[0];
         algo_ = data_[1];
         data_ += 2;
      }
   }

   BlockHeader
   MP7Payload::getHeader()
   {
      LogTrace("L1T") << "Getting header from " << std::hex << std::setw(8) << *data_;

      return BlockHeader(data_++);
   }

  MTF7Payload::MTF7Payload(const uint32_t * data, const uint32_t * end) : Payload(data, end)
  {
    const uint16_t * data16 = reinterpret_cast<const uint16_t*>(data);
    const uint16_t * end16 = reinterpret_cast<const uint16_t*>(end);
    
    if (end16 - data16 < header_size + counter_size + trailer_size) {
      edm::LogError("L1T") << "MTF7 payload smaller than allowed!";
      data_ = end_;
    } else if (
	       ((data16[0] >> 12) != 0x9) || ((data16[1] >> 12) != 0x9) ||
	       ((data16[2] >> 12) != 0x9) || ((data16[3] >> 12) != 0x9) ||
	       ((data16[4] >> 12) != 0xA) || ((data16[5] >> 12) != 0xA) ||
	       ((data16[6] >> 12) != 0xA) || ((data16[7] >> 12) != 0xA) ||
	       ((data16[8] >> 9) != 0b1000000) || ((data16[9] >> 11) != 0) ||
	       ((data16[10] >> 11) != 0) || ((data16[11] >> 11) != 0)) {
         edm::LogError("L1T") << "MTF7 payload has invalid header!";
         data_ = end_;
    } else if (
	       ((data16[12] >> 15) != 0) || ((data16[13] >> 15) != 1) ||
	       ((data16[14] >> 15) != 0) || ((data16[15] >> 15) != 0)) {
      edm::LogError("L1T") << "MTF7 payload has invalid counter block!";
      data_ = end_;
    } else if (
	       false) {
      // TODO: check trailer
    }
  }

  int
  MTF7Payload::count(unsigned int pattern, unsigned int length) const
  {
    unsigned int mask = 0;
    for (; length > 0; length--)
      mask = (mask << 4) | 0xf;
    
    int count = 0;
    for (const auto& p: block_patterns_)
      count += (p & mask) == pattern;
    return count;
  }
  
  bool
  MTF7Payload::valid(unsigned int pattern) const
  {
    for (const auto& p: block_patterns_) {
      if (p == pattern)
	return true;
    }
    return false;
  }
  
  std::auto_ptr<Block>
  MTF7Payload::getBlock()
  {
    if (end_ - data_ < 2)
      return std::auto_ptr<Block>(0);
    
    const uint16_t * data16 = reinterpret_cast<const uint16_t*>(data_);
    const uint16_t * end16 = reinterpret_cast<const uint16_t*>(end_);
    
    // Read in blocks equivalent to 64 bit words, trying to match the
    // pattern of first bits to what is deemed valid.
    std::vector<uint32_t> payload;
    unsigned int pattern = 0;
    unsigned int i = 0;
    for (; i < max_block_length_ and data16 + (i + 1) * 4 <= end16; ++i) {
      for (int j = 0; j < 4; ++j) {
	auto n = i * 4 + j;
	pattern |= (data16[n] >> 15) << n;
	payload.push_back(data16[n]);
      }
      
      if (count(pattern, i + 1) == 1 and valid(pattern))
	break;
    }
    
    if (not valid(pattern)) {
      edm::LogWarning("L1T") << "MTF7 block with unrecognized id 0x" << std::hex << pattern;
      return std::auto_ptr<Block>(0);
    }
    
    data_ += (i + 1) * 2;
    return std::auto_ptr<Block>(new Block(pattern, payload, 0, MTF7));
  }
  
   CTP7Payload::CTP7Payload(const uint32_t * data, const uint32_t * end) : Payload(data, end)
   {
      ++data_;
      size_ = (*data >> size_shift) & size_mask;
      ++data_;
   }

   BlockHeader
   CTP7Payload::getHeader()
   {
      return BlockHeader(data_++, size_);
   }
}
