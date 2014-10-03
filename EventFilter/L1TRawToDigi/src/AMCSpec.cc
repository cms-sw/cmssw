#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/interface/AMCSpec.h"

namespace amc {
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
         return 0x1000;
      return getSize();
   }

   void
   Packet::addPayload(const uint64_t * data, unsigned int size)
   {
      payload_.insert(payload_.end(), data, data + size);
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
   bool
   Header::valid()
   {
      return (getNumberOfAMCs() <= max_amc) && (getFormatVersion() == fov);
   }

   bool
   Packet::parse(const uint64_t *d, unsigned int size)
   {
      // Need at least a header and trailer
      // TODO check if this can be removed
      if (size < 2)
         return false;

      const uint64_t * data = d;
      header_ = Header(data++);

      if (!header_.valid())
         return false;

      if (size < 2 + header_.getNumberOfAMCs())
         return false;

      // initial filling of AMC payloads
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
}
