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
   Header::valid()
   {
      return (getNumberOfAMCs() <= max_amc) && (getFormatVersion() == fov);
   }

   Trailer::Trailer(unsigned int crc, unsigned int blk, unsigned int lv1, unsigned int bx)
   {
      data_ =
         (static_cast<uint64_t>(crc & CRC_mask) << CRC_shift) |
         (static_cast<uint64_t>(blk & BlkNo_mask) << BlkNo_shift) |
         (static_cast<uint64_t>(lv1 & LV1_mask) << LV1_shift) |
         (static_cast<uint64_t>(bx & BX_mask) << BX_shift);
   }

   void
   Packet::add(unsigned int amc_no, unsigned int board, const std::vector<uint64_t>& load)
   {
      edm::LogInfo("AMC") << "Adding board " << board << " with payload size " << load.size()
         << " as payload #" << amc_no;
      // Start by indexing with 1
      payload_.push_back(amc::Packet(amc_no, board, load));
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

      auto block_start = data;
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
         tot_size += amc.blockHeader().getSize();
         tot_nblocks += amc.blockHeader().getBlocks();
         maxblocks = std::max(maxblocks, amc.blockHeader().getBlocks());
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
      auto block_end = data;

      Trailer t(data++);

      std::string check(reinterpret_cast<const char*>(block_start), reinterpret_cast<const char*>(block_end));
      cms::CRC32Calculator crc(check);

      if (crc.checksum() != t.getCRC()) {
         edm::LogWarning("L1T") << "Mismatch in checksums for block 0";
      }

      if (t.getBlock() != 0 ) {
         edm::LogWarning("L1T")
            << "Block trailer mismatch: "
            << "expected block 0, but trailer is for block "
            << t.getBlock();
      }

      // Read in remaining AMC blocks
      for (unsigned int b = 1; b < maxblocks; ++b) {
         block_start = data;

         Header block_h(data++);
         std::vector<amc::BlockHeader> headers;

         for (unsigned int i = 0; i < block_h.getNumberOfAMCs(); ++i)
            headers.push_back(amc::BlockHeader(data++));

         for (const auto& amc: headers) {
            payload_[amc.getAMCNumber() - 1].addPayload(data, amc.getBlockSize());
            data += amc.getBlockSize();
         }

         block_end = data;

         t = Trailer(data++);

         check = std::string(reinterpret_cast<const char*>(block_start), reinterpret_cast<const char*>(block_end));
         crc = cms::CRC32Calculator(check);

         if (crc.checksum() != t.getCRC()) {
            edm::LogWarning("L1T") << "Mismatch in checksums for block " << b;
         }

         if (t.getBlock() != 0 ) {
            edm::LogWarning("L1T")
               << "Block trailer mismatch: "
               << "expected block " << b
               << ", but trailer is for block " << t.getBlock();
         }
      }

      for (auto& amc: payload_) {
         amc.finalize();
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
   Packet::write(const edm::Event& ev, unsigned char * ptr, unsigned int size) const
   {
      if (size < this->size() * 8)
         return false;

      if (size % 8 != 0)
         return false;

      uint64_t * data = reinterpret_cast<uint64_t*>(ptr);

      for (unsigned int b = 0; b < blocks(); ++b) {
         uint64_t * block_start = data;

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

         std::string dstring(reinterpret_cast<char*>(block_start), reinterpret_cast<char*>(data));
         cms::CRC32Calculator crc(dstring);
         *(data++) = Trailer(crc.checksum(), b, ev.id().event(), ev.bunchCrossing()).raw();
      }

      return true;
   }
}
