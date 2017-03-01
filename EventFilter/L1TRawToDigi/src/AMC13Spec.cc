#include <iomanip>
#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/CRC32Calculator.h"

#include "EventFilter/L1TRawToDigi/interface/AMC13Spec.h"

#define EDM_ML_DEBUG 1

namespace amc13 {
   Header::Header(unsigned int namc, unsigned int orbit)
   {
      data_ =
         (static_cast<uint64_t>(namc & nAMC_mask) << nAMC_shift) |
         (static_cast<uint64_t>(orbit & OrN_mask) << OrN_shift) |
         (static_cast<uint64_t>(fov & uFOV_mask) << uFOV_shift);
   }

   bool
   Header::check() const
   {
      return (getNumberOfAMCs() <= max_amc) && (getFormatVersion() == fov);
   }

   Trailer::Trailer(unsigned int blk, unsigned int lv1, unsigned int bx)
   {
      data_ =
         (static_cast<uint64_t>(blk & BlkNo_mask) << BlkNo_shift) |
         (static_cast<uint64_t>(lv1 & LV1_mask) << LV1_shift) |
         (static_cast<uint64_t>(bx & BX_mask) << BX_shift);
   }

   bool
   Trailer::check(unsigned int crc, unsigned int block, unsigned int lv1_id, unsigned int bx) const
   {
      if ((crc != 0 && crc != getCRC()) || block != getBlock() || (lv1_id & LV1_mask) != getLV1ID() || (bx & BX_mask) != getBX()) {
         edm::LogWarning("L1T")
            << "Found AMC13 trailer with:"
            << "\n\tBX " << getBX() << ", LV1 ID " << getLV1ID() << ", block # " << getBlock()
            << ", CRC " << std::hex << std::setw(8) << std::setfill('0') << getCRC() << std::dec
            << "\nBut expected:"
            << "\n\tBX " << (bx & BX_mask) << ", LV1 ID " << (lv1_id & LV1_mask) << ", block # " << block
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

   void
   Packet::add(unsigned int amc_no, unsigned int board, unsigned int lv1id, unsigned int orbit, unsigned int bx, const std::vector<uint64_t>& load)
   {
      edm::LogInfo("AMC") << "Adding board " << board << " with payload size " << load.size()
         << " as payload #" << amc_no;
      // Start by indexing with 1
      payload_.push_back(amc::Packet(amc_no, board, lv1id, orbit, bx, load));
   }

   bool
   Packet::parse(const uint64_t *start, const uint64_t *data, unsigned int size, unsigned int lv1, unsigned int bx, bool legacy_mc, bool mtf7_mode)
   {
      // Need at least a header and trailer
      // TODO check if this can be removed
      if (size < 2) {
         edm::LogError("AMC") << "AMC13 packet size too small";
         return false;
      }

      std::map<int, int> amc_index;

      header_ = Header(data++);

      if (!header_.check()) {
         edm::LogError("AMC")
            << "Invalid header for AMC13 packet: "
            << "format version " << header_.getFormatVersion()
            << ", " << header_.getNumberOfAMCs()
            << " AMC packets";
         return false;
      }

      if (size < 2 + header_.getNumberOfAMCs())
         return false;

      // Initial filling of AMC payloads.  First, get the headers.  The
      // first payload follows afterwards.
      for (unsigned int i = 0; i < header_.getNumberOfAMCs(); ++i) {
         payload_.push_back(amc::Packet(data++));
         amc_index[payload_.back().blockHeader().getAMCNumber()] = i;
      }

      unsigned int tot_size = 0; // total payload size
      unsigned int tot_nblocks = 0; // total blocks of payload
      unsigned int maxblocks = 0; // counting the # of amc13 header/trailers (1 ea per block)

      bool check_crc = false;
      for (const auto& amc: payload_) {
         tot_size += amc.blockHeader().getSize();
         tot_nblocks += amc.blockHeader().getBlocks();
         maxblocks = std::max(maxblocks, amc.blockHeader().getBlocks());

         if (amc.blockHeader().validCRC())
            check_crc = true;
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
         amc.addPayload(data, amc.blockHeader().getBlockSize());
         data += amc.blockHeader().getBlockSize();
      }

      Trailer t(data++);

      int crc = 0;
      if (check_crc) {
         std::string check(reinterpret_cast<const char*>(start), reinterpret_cast<const char*>(data) - 4);
         crc = cms::CRC32Calculator(check).checksum();

         LogDebug("L1T") << "checking data checksum of " << std::hex << crc << std::dec;
      }

      t.check(crc, 0, lv1, bx);

      // Read in remaining AMC blocks
      for (unsigned int b = 1; b < maxblocks; ++b) {
         Header block_h(data++);
         std::vector<amc::BlockHeader> headers;

         for (unsigned int i = 0; i < block_h.getNumberOfAMCs(); ++i)
            headers.push_back(amc::BlockHeader(data++));

         check_crc = false;
         for (const auto& amc: headers) {
            payload_[amc_index[amc.getAMCNumber()]].addPayload(data, amc.getBlockSize());
            data += amc.getBlockSize();

            if (amc.validCRC())
               check_crc = true;
         }

         t = Trailer(data++);

         if (check_crc) {
            std::string check(reinterpret_cast<const char*>(start), reinterpret_cast<const char*>(data) - 4);
            crc = cms::CRC32Calculator(check).checksum();

            LogDebug("L1T") << "checking data checksum of " << std::hex << crc << std::dec;
         } else {
            crc = 0;
         }

         t.check(crc, b, lv1, bx);
      }

      for (auto& amc: payload_) {
	amc.finalize(lv1, bx, legacy_mc, mtf7_mode);
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
         words += amc.header().getSize();
         blocks += amc.blocks();
         maxblocks = std::max(maxblocks, amc.blocks());
      }

      // Size is total amount of words + # of blocks for AMC headers + # of
      // maxblocks for AMC13 block header, trailer
      return words + blocks + maxblocks * 2;
   }

   bool
   Packet::write(const edm::Event& ev, unsigned char * ptr, unsigned int skip, unsigned int size) const
   {
      if (size < this->size() * 8)
         return false;

      if (size % 8 != 0)
         return false;

      uint64_t * data = reinterpret_cast<uint64_t*>(ptr + skip);

      for (unsigned int b = 0; b < blocks(); ++b) {
         // uint64_t * block_start = data;

         std::vector<uint64_t> block_headers;
         std::vector<uint64_t> block_load;
         for (const auto& amc: payload_) {
            edm::LogInfo("AMC")
               << "Considering block " << b
               << " for payload " << amc.blockHeader().getBoardID()
               << " with size " << amc.size()
               << " and " << amc.blocks() << " blocks";
            if (amc.blocks() < b + 1)
               continue;

            block_headers.push_back(amc.blockHeader(b));
            auto words = amc.block(b);
            block_load.insert(block_load.end(), words.begin(), words.end());
         }

         if (b == 0) {
            amc13::Header h(block_headers.size(), ev.orbitNumber());
            edm::LogInfo("AMC")
               << "Writing header for AMC13 packet: "
               << "format version " << h.getFormatVersion()
               << ", " << h.getNumberOfAMCs()
               << " AMC packets, orbit " << h.getOrbitNumber();
         }

         *(data++) = amc13::Header(block_headers.size(), ev.orbitNumber()).raw();

         block_headers.insert(block_headers.end(), block_load.begin(), block_load.end());
         for (const auto& word: block_headers)
            *(data++) = word;

         *data = Trailer(b, ev.id().event(), ev.bunchCrossing()).raw();
         Trailer::writeCRC(reinterpret_cast<uint64_t*>(ptr), data);
      }

      return true;
   }
}
