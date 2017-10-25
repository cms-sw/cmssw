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
   BlockHeader::raw() const
   {
      if (type_ == MP7) {
         LogTrace("L1T") << "Writing MP7 link header";
         return ((id_ & ID_mask) << ID_shift) | ((size_ & size_mask) << size_shift) | ((capID_ & capID_mask) << capID_shift) | ((flags_ & flags_mask) << flags_shift);
      }
      else if (type_ == CTP7) {
         LogTrace("L1T") << "Writing CTP7 link header";
         return flags_;
      }
      // if (type_ == MTF7) {
      //    LogTrace("L1T") << "Writing MTF7 link header";
      //    return ((id_ & ID_mask) << ID_shift) | ((size_ & size_mask) << size_shift) | ((capID_ & capID_mask) << capID_shift);
      // }
      LogTrace("L1T") << "Writing meaningless link header";
      return 0;
   }

   BxBlocks
   Block::getBxBlocks(unsigned int payloadWordsPerBx, bool bxHeader) const
   {
      BxBlocks bxBlocks;

      // For MP7 format
      unsigned int wordsPerBx = payloadWordsPerBx;
      if (bxHeader) {
         ++wordsPerBx;
      }
      // Calculate how many BxBlock objects can be made with the available payload
      unsigned int nBxBlocks = payload_.size() / wordsPerBx;
      for (size_t bxCtr = 0; bxCtr < nBxBlocks; ++bxCtr) {
         size_t startIdx = bxCtr * wordsPerBx;
         auto startBxBlock = payload_.cbegin()+startIdx;
         // Pick the words from the block payload that correspond to the BX and add a BxBlock to the BxBlocks
         if (bxHeader) {
            bxBlocks.emplace_back(startBxBlock, startBxBlock+wordsPerBx);
         } else {
            bxBlocks.emplace_back(bxCtr, nBxBlocks, startBxBlock, startBxBlock+wordsPerBx);
         }
      }

      return bxBlocks;
   }

   std::unique_ptr<Block>
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

      auto res = std::unique_ptr<Block>(new Block(header, data_, data_ + header.getSize()));
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
  
  std::unique_ptr<Block>
  MTF7Payload::getBlock()
  {
    if (end_ - data_ < 2)
      return std::auto_ptr<Block>(nullptr);
    
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
      return std::auto_ptr<Block>(nullptr);
    }
    
    data_ += (i + 1) * 2;
    return std::unique_ptr<Block>(new Block(pattern, payload, 0, MTF7));
  }
  
   CTP7Payload::CTP7Payload(const uint32_t * data, const uint32_t * end, amc::Header amcHeader) : Payload(data, end), amcHeader_(amcHeader)
   {
      if (not (*data_ == 0xA110CA7E) ) {
        edm::LogError("L1T") << "CTP7 block with invalid header:" << std::hex << *data_;
      }
      ++data_;
      bx_per_l1a_ = (*data_ >> 16) & 0xff;
      calo_bxid_ = *data_ & 0xfff;
      capId_ = 0;
      if (bx_per_l1a_ > 1) {
        edm::LogInfo("L1T") << "CTP7 block with multiple bunch crossings:" << bx_per_l1a_;
      }
      algo_ = amcHeader_.getUserData();
      infra_ = 0;
      ++data_;
   }

   BlockHeader
   CTP7Payload::getHeader()
   {
      // only one block type, use dummy id
      unsigned blockId = 0;
      // CTP7 header contains number of BX in payload and the bunch crossing ID
      // Not sure how to map to generic BlockHeader variables, so just packing
      // it all in flags variable
      unsigned blockFlags = ((bx_per_l1a_&0xf)<<16) | (calo_bxid_&0xfff);
      unsigned blockSize = 192;
      return BlockHeader(blockId, blockSize, capId_, blockFlags, CTP7);
   }

   std::unique_ptr<Block>
   CTP7Payload::getBlock()
   {
      if (end_ - data_ < getHeaderSize()) {
         LogDebug("L1T") << "Reached end of payload";
         return std::auto_ptr<Block>();
      }
      if ( capId_ > bx_per_l1a_ ) {
        edm::LogWarning("L1T") << "CTP7 with more bunch crossings than expected";
      }

      auto header = getHeader();

      if (end_ - data_ < header.getSize()) {
         edm::LogError("L1T")
            << "Expecting a block size of " << header.getSize()
            << " but only " << (end_ - data_) << " words remaining";
         return std::auto_ptr<Block>();
      }

      LogTrace("L1T") << "Creating block with size " << header.getSize();

      auto res = std::unique_ptr<Block>(new Block(header, data_, data_ + header.getSize()));
      data_ += header.getSize();
      capId_++;
      return res;
   }
}
