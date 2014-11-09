#ifndef AMC_SPEC__h
#define AMC_SPEC__h

#include <memory>
#include <vector>
#include <stdint.h>

namespace edm {
   class Event;
}

namespace amc {
   static const unsigned int split_block_size = 0x1000;

   class Header {
      public:
         Header(const uint64_t *data) : data_(data[0]) {};
         // size is the total size of the AMC payload, not just of the
         // block
         Header(unsigned int amc_no, unsigned int board_id, unsigned int size, unsigned int block=0);

         operator uint64_t() const { return data_; };

         inline uint64_t raw() const { return data_; };

         unsigned int getBlocks() const;
         unsigned int getBlockSize() const;

         inline unsigned int getAMCNumber() const { return (data_ >> AmcNo_shift) & AmcNo_mask; };
         inline unsigned int getBoardID() const { return (data_ >> BoardID_shift) & BoardID_mask; };
         inline unsigned int getSize() const { return (data_ >> Size_shift) & Size_mask; };
         inline unsigned int getMore() const { return (data_ >> More_bit_shift) & 1; };
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
         Packet(unsigned int amc, unsigned int board, const std::vector<uint64_t>& load);

         void addPayload(const uint64_t*, unsigned int);

         std::vector<uint64_t> block(unsigned int id) const;
         std::unique_ptr<uint64_t[]> data();
         Header header(unsigned int block=0) const { return header_; };

         inline unsigned int blocks() const { return header_.getBlocks(); };
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
         Header(unsigned int namc, unsigned int orbit);

         bool valid();

         inline uint64_t raw() const { return data_; };

         inline unsigned int getFormatVersion() const { return (data_ >> uFOV_shift) & uFOV_mask; };
         inline unsigned int getNumberOfAMCs() const { return (data_ >> nAMC_shift) & nAMC_mask; };
         inline unsigned int getOrbitNumber() const { return (data_ >> OrN_shift) & OrN_mask; };

      private:
         static const unsigned int uFOV_shift = 60;
         static const unsigned int uFOV_mask = 0xf;
         static const unsigned int nAMC_shift = 52;
         static const unsigned int nAMC_mask = 0xf;
         static const unsigned int OrN_shift = 4;
         static const unsigned int OrN_mask = 0xffffffff;

         static const unsigned int fov = 1;
         static const unsigned int max_amc = 12;

         uint64_t data_;
   };

   class Trailer {
      public:
         Trailer(const uint64_t *data) : data_(data[0]) {};
         Trailer(unsigned int crc, unsigned int blk, unsigned int lv1, unsigned int bx);

         inline unsigned int getCRC() const { return (data_ >> CRC_shift) & CRC_mask; };
         inline unsigned int getBlock() const { return (data_ >> BlkNo_shift) & BlkNo_mask; };
         inline unsigned int getLV1ID() const { return (data_ >> LV1_shift) & LV1_mask; };
         inline unsigned int getBX() const { return (data_ >> BX_shift) & BX_mask; };

         uint64_t raw() const { return data_; };

      private:
         static const unsigned int CRC_shift = 32;
         static const unsigned int CRC_mask = 0xffffffff;
         static const unsigned int BlkNo_shift = 20;
         static const unsigned int BlkNo_mask = 0xff;
         static const unsigned int LV1_shift = 12;
         static const unsigned int LV1_mask = 0xff;
         static const unsigned int BX_shift = 0;
         static const unsigned int BX_mask = 0xfff;

         uint64_t data_;
   };

   class Packet {
      public:
         Packet() {};

         unsigned int blocks() const;
         unsigned int size() const;

         void add(unsigned int board, const std::vector<uint64_t>& load);
         bool parse(const uint64_t*, unsigned int);
         bool write(const edm::Event& ev, unsigned char * ptr, unsigned int size) const;

         inline std::vector<amc::Packet> payload() const { return payload_; };

      private:
         Header header_;
         std::vector<amc::Packet> payload_;
   };
}

#endif
