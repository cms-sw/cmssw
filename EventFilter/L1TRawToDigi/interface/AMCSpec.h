#ifndef AMC_SPEC__h
#define AMC_SPEC__h

#include <memory>
#include <vector>
#include <stdint.h>

namespace amc {
   class Header {
      public:
         Header(const uint64_t *data) : data_(data[0]) {};

         unsigned int getBlocks() const;
         unsigned int getBlockSize() const;

         inline unsigned int getAMCNumber() const { return (data_ >> AmcNo_shift) & AmcNo_mask; };
         inline unsigned int getBoardID() const { return (data_ >> BoardID_shift) & BoardID_mask; };
         inline unsigned int getSize() const { return (data_ >> Size_shift) & Size_mask; };
         inline unsigned int getMore() const { return (data_ >> Length_bit_shift) & 1; };
         inline unsigned int getSegmented() const { return (data_ >> Segmented_bit_shift) & 1; };

      private:
         static const unsigned int Size_shift = 32;
         static const unsigned int Size_mask = 0xffffff;
         static const unsigned int BlkNo_shift = 20;
         static const unsigned int BlkNo_mask = 0xff;
         static const unsigned int AmcNo_shift = 16;
         static const unsigned int AmcNo_mask = 0xf;
         static const unsigned int BoardID_shift = 0;
         static const unsigned int BoardID_mask = 0xffff;

         static const unsigned int Length_bit_shift = 62;
         static const unsigned int More_bit_shift = 61;
         static const unsigned int Segmented_bit_shift = 60;
         static const unsigned int Enabled_bit_shift = 59;
         static const unsigned int Present_bit_shift = 58;
         static const unsigned int Valid_bit_shift = 57;
         static const unsigned int CRC_bit_shift = 56;

         uint64_t data_;
   };

   class Packet {
      public:
         Packet(const uint64_t* d) : header_(d) {};

         void addPayload(const uint64_t*, unsigned int);
         inline Header header() const { return header_; };
         std::unique_ptr<uint64_t[]> data();
         inline unsigned int size() const { return header_.getSize(); };

      private:
         Header header_;
         std::vector<uint64_t> payload_;
   };
}

namespace amc13 {
   class Header {
      public:
         Header() : data_(0) {};
         Header(const uint64_t *data) : data_(data[0]) {};

         bool valid();

         inline unsigned int getFormatVersion() { return (data_ << uFOV_shift) & uFOV_mask; };
         inline unsigned int getNumberOfAMCs() { return (data_ << nAMC_shift) & nAMC_mask; };
         inline unsigned int getOrbitNumber() { return (data_ << OrN_shift) & OrN_mask; };

      private:
         static const unsigned int uFOV_shift = 60;
         static const unsigned int uFOV_mask = 0xf;
         static const unsigned int nAMC_shift = 42;
         static const unsigned int nAMC_mask = 0xf;
         static const unsigned int OrN_shift = 4;
         static const unsigned int OrN_mask = 0xffffffff;

         static const unsigned int fov = 1;
         static const unsigned int max_amc = 12;

         uint64_t data_;
   };

   class Packet {
      public:
         Packet() {};

         bool parse(const uint64_t*, unsigned int);
         inline std::vector<amc::Packet> payload() const { return payload_; };

      private:
         Header header_;
         std::vector<amc::Packet> payload_;
   };
}

#endif
