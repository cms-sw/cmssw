#ifndef EventFilter_L1TRawToDigi_BxBlock_h
#define EventFilter_L1TRawToDigi_BxBlock_h

#include <memory>
#include <vector>
#include <cmath>

namespace l1t {
   class BxBlockHeader {
      public:
         BxBlockHeader() : id_(0), totalBx_(0), flags_(0) {};
         BxBlockHeader(unsigned int id, unsigned int totalBx, unsigned int flags=0) : id_(id), totalBx_(totalBx), flags_(flags) {};
         // Create a BX header: everything is contained in the raw uint32
         BxBlockHeader(const uint32_t raw) : id_(((raw >> id_shift) & id_mask) / n_words)
                                      , totalBx_(((raw >> totalBx_shift) & totalBx_mask) / n_words)
                                      , flags_((raw >> flags_shift) & flags_mask) {};

         bool operator<(const BxBlockHeader& o) const { return getBx() < o.getBx(); };

         inline int getBx() const { return (int)id_ - (int)std::floor(totalBx_/2.); };
         inline unsigned int getId() const { return id_; };
         inline unsigned int getTotalBx() const { return totalBx_; };
         inline unsigned int getFlags() const { return flags_; };

         inline uint32_t raw() const { return (((id_ & id_mask) << id_shift) * n_words)
                                     | (((totalBx_ & totalBx_mask) << totalBx_shift) * n_words)
                                     | ((flags_ & flags_mask) << flags_shift); };

      private:
         static const unsigned int n_words = 6; // every link transmits 6 32 bit words per bx
         static const unsigned int id_shift = 24;
         static const unsigned int id_mask = 0xff;
         static const unsigned int totalBx_shift = 16;
         static const unsigned int totalBx_mask = 0xff;
         static const unsigned int flags_shift = 0;
         static const unsigned int flags_mask = 0xffff;

         unsigned int id_;
         unsigned int totalBx_;
         unsigned int flags_;
   };

   class BxBlock {
      public:
         BxBlock(std::vector<uint32_t>::const_iterator bx_start, std::vector<uint32_t>::const_iterator bx_end) :
            header_(*bx_start), payload_(bx_start+1, bx_end) {};
         BxBlock(const BxBlockHeader& h, std::vector<uint32_t>::const_iterator payload_start, std::vector<uint32_t>::const_iterator payload_end) :
            header_(h), payload_(payload_start, payload_end) {};
         BxBlock(unsigned int id, unsigned int totalBx, std::vector<uint32_t>::const_iterator payload_start, std::vector<uint32_t>::const_iterator payload_end, unsigned int flags=0) :
            header_(id, totalBx, flags), payload_(payload_start, payload_end) {};
         BxBlock(unsigned int id, unsigned int totalBx, const std::vector<uint32_t>& payload, unsigned int flags=0) :
            header_(id, totalBx, flags), payload_(payload) {};
         ~BxBlock() {};

         bool operator<(const BxBlock& o) const { return header() < o.header(); };

         inline unsigned int getSize() const { return payload_.size(); };

         BxBlockHeader header() const { return header_; };
         std::vector<uint32_t> payload() const { return payload_; };

      private:
         BxBlockHeader header_;
         std::vector<uint32_t> payload_;
   };

   typedef std::vector<BxBlock> BxBlocks;
}

#endif
