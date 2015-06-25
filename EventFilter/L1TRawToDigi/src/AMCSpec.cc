#include <iomanip>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/CRC32Calculator.h"

#include "EventFilter/L1TRawToDigi/interface/AMCSpec.h"

#define EDM_ML_DEBUG 1

namespace amc {
   BlockHeader::BlockHeader(unsigned int amc_no, unsigned int board_id, unsigned int size, unsigned int block)
   {
      // Determine size
      unsigned int max_block_no = 0;
      if (size >= 0x13ff)
         max_block_no = (size - 1023) / 4096;

      if (block != max_block_no)
         size = split_block_size;
      else if (block != 0)
         size -= split_block_size * max_block_no;

      data_ =
         (static_cast<uint64_t>(size & Size_mask) << Size_shift) |
         (static_cast<uint64_t>(block & BlkNo_mask) << BlkNo_shift) |
         (static_cast<uint64_t>(amc_no & AmcNo_mask) << AmcNo_shift) |
         (static_cast<uint64_t>(board_id & BoardID_mask) << BoardID_shift) |
         (1llu << Enabled_bit_shift) |
         (1llu << Present_bit_shift);

      if (block == getBlocks() - 1) {
         // Last block
         data_ |=
            (1llu << CRC_bit_shift) |
            (1llu << Valid_bit_shift) |
            (1llu << Length_bit_shift);
      }

      if (block == 0 && getBlocks() == 1) {
         // Bits already zeroed - only one block
      } else if (block == 0) {
         // First of many blocks
         data_ |= 1llu << More_bit_shift;
      } else if (block == getBlocks() - 1) {
         // Last of many blocks
         data_ |= 1llu << Segmented_bit_shift;
      } else {
         // Intermediate of many blocks
         data_ |= (1llu << More_bit_shift) | (1llu << Segmented_bit_shift);
      }
   }

   unsigned int
   BlockHeader::getBlocks() const
   {
      // The first block of a segmented event has a size of 1023, all
      // following have a max size of 4096.  Segmentation only happens
      // for AMC payloads >= 0x13ff 64 bit words.
      unsigned int size = getSize();
      if (size >= 0x13ff)
         return (size - 1023) / 4096 + 1;
      return 1;
   }

   unsigned int
   BlockHeader::getBlockSize() const
   {
      // More and not Segmented means the first of multiple blocks.  For
      // these, getSize() returns the total size of the AMC packet, not the
      // size of the first block.
      if (getMore() && !getSegmented())
         return split_block_size;
      return getSize();
   }

   Header::Header(unsigned int amc_no, unsigned int lv1_id, unsigned int bx_id, unsigned int size,
         unsigned int or_n, unsigned int board_id, unsigned int user) :
      data0_(
            (uint64_t(amc_no & AmcNo_mask) << AmcNo_shift) |
            (uint64_t(lv1_id & LV1ID_mask) << LV1ID_shift) |
            (uint64_t(bx_id & BX_mask) << BX_shift) |
            (uint64_t(size & Size_mask) << Size_shift)
      ),
      data1_(
            (uint64_t(or_n & OrN_mask) << OrN_shift) |
            (uint64_t(board_id & BoardID_mask) << BoardID_shift) |
            (uint64_t(user & User_mask) << User_shift)
      )
   {
   }

   Trailer::Trailer(unsigned int crc, unsigned int lv1_id, unsigned int size) :
      data_(
            (uint64_t(crc & CRC_mask) << CRC_shift) |
            (uint64_t(lv1_id & LV1ID_mask) << LV1ID_shift) |
            (uint64_t(size & Size_mask) << Size_shift)
      )
   {
   }

   bool
   Trailer::check(unsigned int crc, unsigned int lv1_id, unsigned int size) const
   {
      if (crc != getCRC() || size != getSize() || (lv1_id & LV1ID_mask) != getLV1ID()) {
         edm::LogWarning("L1T")
            << "Found AMC trailer with:"
            << "\n\tLV1 ID " << getLV1ID() << ", size " << getSize()
            << ", CRC " << std::hex << std::setw(8) << std::setfill('0') << getCRC() << std::dec
            << "\nBut expected:"
            << "\n\tLV1 ID " << (lv1_id & LV1ID_mask) << ", size " << size
            << ", CRC " << std::hex << std::setw(8) << std::setfill('0') << crc;
         return false;
      }
      return true;
   }

   void
   Trailer::writeCRC(const uint64_t *start, uint64_t *end)
   {
      std::string dstring(reinterpret_cast<const char*>(start), reinterpret_cast<const char*>(end) + 4);
      auto crc = cms::CRC32Calculator(dstring).checksum();

      *end = ((*end) & ~(uint64_t(CRC_mask) << CRC_shift)) | (static_cast<uint64_t>(crc & CRC_mask) << CRC_shift);
   }

   Packet::Packet(unsigned int amc, unsigned int board, unsigned int lv1id, unsigned int orbit, unsigned int bx, const std::vector<uint64_t>& load) :
      block_header_(amc, board, load.size() + 3), // add 3 words for header (2) and trailer (1)
      header_(amc, lv1id, bx, load.size() + 3, orbit, board, 0),
      trailer_(0, lv1id, load.size() + 3)
   {
      auto hdata = header_.raw();
      payload_.reserve(load.size() + 3);
      payload_.insert(payload_.end(), hdata.begin(), hdata.end());
      payload_.insert(payload_.end(), load.begin(), load.end());
      payload_.insert(payload_.end(), trailer_.raw());

      auto ptr = payload_.data();
      Trailer::writeCRC(ptr, ptr + payload_.size() - 1);
   }

   void
   Packet::addPayload(const uint64_t * data, unsigned int size)
   {
      payload_.insert(payload_.end(), data, data + size);
   }

   void
   Packet::finalize(unsigned int lv1, unsigned int bx, bool legacy_mc)
   {
      if (legacy_mc) {
         header_ = Header(block_header_.getAMCNumber(), lv1, bx, block_header_.getSize(), 0, block_header_.getBoardID(), 0);

         payload_.insert(payload_.begin(), {0, 0});
         payload_.insert(payload_.end(), {0});
      } else {
         header_ = Header(payload_.data());
         trailer_ = Trailer(&payload_.back());

         std::string check(reinterpret_cast<const char*>(payload_.data()), payload_.size() * 8 - 4);
         auto crc = cms::CRC32Calculator(check).checksum();

         trailer_.check(crc, lv1, header_.getSize());
      }
   }

   std::vector<uint64_t>
   Packet::block(unsigned int id) const
   {
      if (id == 0 and id == block_header_.getBlocks() - 1) {
         return payload_;
      } else if (id == block_header_.getBlocks() - 1) {
         return std::vector<uint64_t>(payload_.begin() + id * split_block_size, payload_.end());
      } else {
         return std::vector<uint64_t>(payload_.begin() + id * split_block_size, payload_.begin() + (id + 1) * split_block_size);
      }
   }

   std::unique_ptr<uint64_t[]>
   Packet::data()
   {
      // Remove 3 words: 2 for the header, 1 for the trailer
      std::unique_ptr<uint64_t[]> res(new uint64_t[payload_.size() - 3]);
      for (unsigned int i = 0; i < payload_.size() - 3; ++i)
         res.get()[i] = payload_[i + 2];
      return res;
   }
}
