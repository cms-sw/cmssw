#include <iomanip>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/L1TRawToDigi/interface/Block.h"

#define EDM_ML_DEBUG 1

namespace l1t {
   uint32_t
   BlockHeader::raw(block_t type) const
   {
      if (type_ == MP7) {
         LogTrace("L1T") << "Writing MP7 link header";
         return ((id_ & ID_mask) << ID_shift) | ((size_ & size_mask) << size_shift) | ((capID_ & capID_mask) << capID_shift);
      }
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
