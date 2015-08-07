#ifndef Block_h
#define Block_h

#include <memory>
#include <vector>

#include "EventFilter/L1TRawToDigi/interface/AMCSpec.h"

namespace l1t {
   enum block_t { MP7 = 0, CTP7 };

   class BlockHeader {
      public:
         BlockHeader(unsigned int id, unsigned int size, unsigned int capID=0, block_t type=MP7) : id_(id), size_(size), capID_(capID), type_(type) {};
         // Create a MP7 block header: everything is contained in the raw uint32
         BlockHeader(const uint32_t *data) : id_((data[0] >> ID_shift) & ID_mask), size_((data[0] >> size_shift) & size_mask), capID_((data[0] >> capID_shift) & capID_mask), type_(MP7) {};
         // Create a CTP7 block header: size is contained in the general CTP7 header
         BlockHeader(const uint32_t *data, unsigned int size) : id_((data[0] >> CTP7_shift) & CTP7_mask), size_(size), capID_(0), type_(CTP7) {};

         bool operator<(const BlockHeader& o) const { return getID() < o.getID(); };

         unsigned int getID() const { return id_; };
         unsigned int getSize() const { return size_; };
         unsigned int getCapID() const { return capID_; };
         block_t getType() const { return type_; };

         uint32_t raw(block_t type=MP7) const;

      private:
         static const unsigned int CTP7_shift = 0;
         static const unsigned int CTP7_mask = 0xffff;
         static const unsigned int ID_shift = 24;
         static const unsigned int ID_mask = 0xff;
         static const unsigned int size_shift = 16;
         static const unsigned int size_mask = 0xff;
         static const unsigned int capID_shift = 8;
         static const unsigned int capID_mask = 0xff;

         unsigned int id_;
         unsigned int size_;
         unsigned int capID_;
         block_t type_;
   };

   class Block {
      public:
         Block(const BlockHeader& h, const uint32_t * payload_start, const uint32_t * payload_end) :
            header_(h), payload_(payload_start, payload_end) {};
         Block(unsigned int id, const std::vector<uint32_t>& payload, unsigned int capID=0, block_t type=MP7) :
            header_(id, payload.size(), capID, type), payload_(payload) {};

         bool operator<(const Block& o) const { return header() < o.header(); };

         inline unsigned int getSize() const { return payload_.size() + 1; };

         BlockHeader header() const { return header_; };
         std::vector<uint32_t> payload() const { return payload_; };

         void amc(const amc::Header& h) { amc_ = h; };
         amc::Header amc() const { return amc_; };

      private:
         BlockHeader header_;
         amc::Header amc_;
         std::vector<uint32_t> payload_;
   };

   typedef std::vector<Block> Blocks;

   class Payload {
      public:
         Payload(const uint32_t * data, const uint32_t * end) : data_(data), end_(end), algo_(0), infra_(0) {};

         virtual unsigned getAlgorithmFWVersion() const { return algo_; };
         virtual unsigned getInfrastructureFWVersion() const { return infra_; };
         virtual unsigned getHeaderSize() const = 0;
         // Read header from data_ and advance data_ to point behind the
         // header.  Called by getBlock(), which also checks that data_ !=
         // end_ before calling (assumes size of one 32 bit word).
         virtual BlockHeader getHeader() = 0;
         std::auto_ptr<Block> getBlock();
      protected:
         const uint32_t * data_;
         const uint32_t * end_;

         unsigned algo_;
         unsigned infra_;
   };

   class MP7Payload : public Payload {
      public:
         MP7Payload(const uint32_t * data, const uint32_t * end, bool legacy_mc=false);
         virtual unsigned getHeaderSize() const override { return 1; };
         virtual BlockHeader getHeader() override;
   };

   class CTP7Payload : public Payload {
      public:
         CTP7Payload(const uint32_t * data, const uint32_t * end);
         virtual unsigned getHeaderSize() const override { return 2; };
         virtual BlockHeader getHeader() override;
      private:
         // FIXME check values
         static const unsigned int size_mask = 0xff;
         static const unsigned int size_shift = 16;

         unsigned size_;
   };
}

#endif
