#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/interface/AMCSpec.h"

#define EDM_ML_DEBUG 1

namespace amc {
   Header::Header(unsigned int amc_no, unsigned int board_id, unsigned int size, unsigned int block)
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
   Header::getBlocks() const
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
   Header::getBlockSize() const
   {
      // More and not Segmented means the first of multiple blocks.  For
      // these, getSize() returns the total size of the AMC packet, not the
      // size of the first block.
      if (getMore() && !getSegmented())
         return split_block_size;
      return getSize();
   }

   Packet::Packet(unsigned int amc, unsigned int board, const std::vector<uint64_t>& load) :
      header_(amc, board, load.size()),
      payload_(load)
   {
   }

   void
   Packet::addPayload(const uint64_t * data, unsigned int size)
   {
      payload_.insert(payload_.end(), data, data + size);
   }

   std::vector<uint64_t>
   Packet::block(unsigned int id) const
   {
      if (id == 0 and id == header_.getBlocks() - 1) {
         return payload_;
      } else if (id == header_.getBlocks() - 1) {
         return std::vector<uint64_t>(payload_.begin() + id * split_block_size, payload_.end());
      } else {
         return std::vector<uint64_t>(payload_.begin() + id * split_block_size, payload_.begin() + (id + 1) * split_block_size);
      }
   }

   std::unique_ptr<uint64_t[]>
   Packet::data()
   {
      std::unique_ptr<uint64_t[]> res(new uint64_t[payload_.size()]);
      for (unsigned int i = 0; i < payload_.size(); ++i)
         res.get()[i] = payload_[i];
      return res;
   }
}

namespace amc13 {
   Header::Header(unsigned int namc, unsigned int orbit)
   {
      data_ =
         (static_cast<uint64_t>(namc & nAMC_mask) << nAMC_shift) |
         (static_cast<uint64_t>(orbit & OrN_mask) << OrN_shift) |
         (static_cast<uint64_t>(fov & uFOV_mask) << uFOV_shift);
   }

   bool
   Header::valid()
   {
      return (getNumberOfAMCs() <= max_amc) && (getFormatVersion() == fov);
   }

   void
   Packet::add(unsigned int board, const std::vector<uint64_t>& load)
   {
      edm::LogInfo("AMC") << "Adding board " << board << " with payload size " << load.size();
      payload_.push_back(amc::Packet(payload_.size(), board, load));
   }

   bool
   Packet::parse(const uint64_t *data, unsigned int size)
   {
      // Need at least a header and trailer
      // TODO check if this can be removed
      if (size < 2) {
         edm::LogError("AMC") << "AMC13 packet size too small";
         return false;
      }

      /* const uint64_t * data = d; */
      header_ = Header(data++);

      if (!header_.valid()) {
         edm::LogError("AMC")
            << "Invalid header for AMC13 packet: "
            << "format version " << header_.getFormatVersion()
            << ", " << header_.getNumberOfAMCs()
            << " AMC packets, orbit " << header_.getOrbitNumber();
         return false;
      }

      if (size < 2 + header_.getNumberOfAMCs())
         return false;

      // Initial filling of AMC payloads.  First, get the headers.  The
      // first payload follows afterwards.
      for (unsigned int i = 0; i < header_.getNumberOfAMCs(); ++i) {
         payload_.push_back(amc::Packet(data++));
      }

      unsigned int tot_size = 0; // total payload size
      unsigned int tot_nblocks = 0; // total blocks of payload
      unsigned int maxblocks = 0; // counting the # of amc13 header/trailers (1 ea per block)

      for (const auto& amc: payload_) {
         tot_size += amc.header().getSize();
         tot_nblocks += amc.header().getBlocks();
         maxblocks = std::max(maxblocks, amc.header().getBlocks());
      }

      unsigned int words = tot_size + // payload size
                           tot_nblocks + // AMC headers
                           2 * maxblocks; // AMC13 headers

      if (size < words) {
         edm::LogError("L1T")
            << "Encountered AMC 13 packet with "
            << size << " words, "
            << "but expected "
            << words << " words: "
            << tot_size << " payload words, "
            << tot_nblocks << " AMC header words, and 2 AMC 13 header words.";
         return false;
      }

      // Read in the first AMC block and append the payload to the
      // corresponding AMC packet.
      for (auto& amc: payload_) {
         amc.addPayload(data, amc.header().getBlockSize());
         data += amc.header().getBlockSize();
      }

      // Skip trailer
      data++;

      // Read in remaining AMC blocks
      for (unsigned int b = 1; b < maxblocks; ++b) {
         Header block_h(data++);
         std::vector<amc::Header> headers;

         for (unsigned int i = 0; i < block_h.getNumberOfAMCs(); ++i)
            headers.push_back(amc::Header(data++));

         for (const auto& amc: headers) {
            payload_[amc.getAMCNumber()].addPayload(data, amc.getBlockSize());
            data += amc.getBlockSize();
         }

         // Skip trailer
         data++;
      }

      return true;
   }

   unsigned int
   Packet::blocks() const
   {
      unsigned int maxblocks = 0;

      for (const auto& amc: payload_)
         maxblocks = std::max(maxblocks, amc.blocks());

      return maxblocks;
   }

   unsigned int
   Packet::size() const
   {
      unsigned int words = 0;
      unsigned int blocks = 0;
      unsigned int maxblocks = 0;

      for (const auto& amc: payload_) {
         words += amc.size();
         blocks += amc.blocks();
         maxblocks = std::max(maxblocks, amc.blocks());
      }

      // Size is total amount of words + # of blocks for AMC headers + # of
      // maxblocks for AMC13 block header, trailer
      return words + blocks + maxblocks * 2;
   }

   bool
   Packet::write(unsigned int orbit, unsigned char * ptr, unsigned int size) const
   {
      if (size < this->size() * 8)
         return false;

      if (size % 8 != 0)
         return false;

      uint64_t * data = reinterpret_cast<uint64_t*>(ptr);
      /* uint64_t * end = data + (size / 8); */

      for (unsigned int b = 0; b < blocks(); ++b) {
         std::vector<uint64_t> block_headers;
         std::vector<uint64_t> block_load;
         for (const auto& amc: payload_) {
            edm::LogInfo("AMC")
               << "Considering block " << b
               << " for payload " << amc.header().getBoardID()
               << " with size " << amc.size()
               << " and " << amc.blocks() << " blocks";
            if (amc.blocks() < b + 1)
               continue;

            block_headers.push_back(amc.header(b));
            auto words = amc.block(b);
            block_load.insert(block_load.end(), words.begin(), words.end());
         }

         if (b == 0) {
            amc13::Header h(block_headers.size(), orbit);
            edm::LogInfo("AMC")
               << "Writing header for AMC13 packet: "
               << "format version " << h.getFormatVersion()
               << ", " << h.getNumberOfAMCs()
               << " AMC packets, orbit " << h.getOrbitNumber();
         }

         *(data++) = amc13::Header(block_headers.size(), orbit).raw();

         block_headers.insert(block_headers.end(), block_load.begin(), block_load.end());
         for (const auto& word: block_headers)
            *(data++) = word;

         // FIXME skip trailer
         ++data;
      }

      return true;
   }
}
