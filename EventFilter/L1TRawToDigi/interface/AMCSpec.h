#ifndef AMC_SPEC__h
#define AMC_SPEC__h

#include <memory>
#include <vector>
#include <stdint.h>

namespace amc {
   static const unsigned int split_block_size = 0x1000;

   // The AMC header within an AMC13 payload block.  Should optimally only
   // be used when packing/unpacking AMC payloads into AMC13 blocks.
   class BlockHeader {
      public:
         BlockHeader() : data_(0) {};
         BlockHeader(const uint64_t *data) : data_(data[0]) {};
         // size is the total size of the AMC payload, not just of the
         // block
         BlockHeader(unsigned int amc_no, unsigned int board_id, unsigned int size, unsigned int block=0);

         operator uint64_t() const { return data_; };

         inline uint64_t raw() const { return data_; };

         unsigned int getBlocks() const;
         unsigned int getBlockSize() const;

         inline unsigned int getAMCNumber() const { return (data_ >> AmcNo_shift) & AmcNo_mask; };
         inline unsigned int getBoardID() const { return (data_ >> BoardID_shift) & BoardID_mask; };
         inline unsigned int getSize() const { return (data_ >> Size_shift) & Size_mask; };
         inline unsigned int getMore() const { return (data_ >> More_bit_shift) & 1; };
         inline unsigned int getSegmented() const { return (data_ >> Segmented_bit_shift) & 1; };

         inline unsigned int validCRC() const { return (data_ >> CRC_bit_shift) & 1; };

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

   // The actual header attached to the AMC payload, also contained in the
   // AMC payload of an AMC13 packet/block.
   class Header {
      public:
         Header() : data0_(0), data1_(0) {};
         Header(const uint64_t *data) : data0_(data[0]), data1_(data[1]) {};
         Header(unsigned int amc_no, unsigned int lv1_id, unsigned int bx_id, unsigned int size,
               unsigned int or_n, unsigned int board_id, unsigned int user);

         inline unsigned int getAMCNumber() const { return (data0_ >> AmcNo_shift) & AmcNo_mask; };
         inline unsigned int getBoardID() const { return (data1_ >> BoardID_shift) & BoardID_mask; };
         inline unsigned int getLV1ID() const { return (data0_ >> LV1ID_shift) & LV1ID_mask; };
         inline unsigned int getBX() const { return (data0_ >> BX_shift) & BX_mask; };
         inline unsigned int getOrbitNumber() const { return (data1_ >> OrN_shift) & OrN_mask; };
         inline unsigned int getSize() const { return (data0_ >> Size_shift) & Size_mask; };
         inline unsigned int getUserData() const { return (data1_ >> User_shift) & User_mask; };

         std::vector<uint64_t> raw() const { return {data0_, data1_}; };

      private:
         static const unsigned int Size_shift = 0;
         static const unsigned int Size_mask = 0xfffff;
         static const unsigned int BX_shift = 20;
         static const unsigned int BX_mask = 0xfff;
         static const unsigned int LV1ID_shift = 32;
         static const unsigned int LV1ID_mask = 0xffffff;
         static const unsigned int AmcNo_shift = 56;
         static const unsigned int AmcNo_mask = 0xf;

         static const unsigned int BoardID_shift = 0;
         static const unsigned int BoardID_mask = 0xffff;
         static const unsigned int OrN_shift = 16;
         static const unsigned int OrN_mask = 0xffff;
         static const unsigned int User_shift = 32;
         static const unsigned int User_mask = 0xffffffff;

         uint64_t data0_;
         uint64_t data1_;
   };

   class Trailer {
      public:
         Trailer() : data_(0) {};
         Trailer(const uint64_t *data) : data_(data[0]) {};
         Trailer(unsigned int crc, unsigned int lv1_id, unsigned int size);

         inline unsigned int getCRC() const { return (data_ >> CRC_shift) & CRC_mask; };
         inline unsigned int getLV1ID() const { return (data_ >> LV1ID_shift) & LV1ID_mask; };
         inline unsigned int getSize() const { return (data_ >> Size_shift) & Size_mask; };

         uint64_t raw() const { return data_; }
         bool check(unsigned int crc, unsigned int lv1_id, unsigned int size, bool mtf7_mode=false) const;

         static void writeCRC(const uint64_t *start, uint64_t *end);

      private:
         static const unsigned int Size_shift = 0;
         static const unsigned int Size_mask = 0xfffff;
         static const unsigned int LV1ID_shift = 24;
         static const unsigned int LV1ID_mask = 0xff;
         static const unsigned int CRC_shift = 32;
         static const unsigned int CRC_mask = 0xffffffff;

         uint64_t data_;
   };

   class Packet {
      public:
         Packet(const uint64_t* d) : block_header_(d) {};
         Packet(unsigned int amc, unsigned int board, unsigned int lv1id, unsigned int orbit, unsigned int bx, const std::vector<uint64_t>& load);

         // Add payload fragment from an AMC13 block to the AMC packet
         void addPayload(const uint64_t*, unsigned int);
         // To be called after the last payload addition.  Removes header
         // and trailer from the actual paylod.  Also performs
         // cross-checks for data consistency.
         void finalize(unsigned int lv1, unsigned int bx, bool legacy_mc=false, bool mtf7_mode=false);

         std::vector<uint64_t> block(unsigned int id) const;
         std::unique_ptr<uint64_t[]> data();
         BlockHeader blockHeader(unsigned int block=0) const { return block_header_; };
         Header header() const { return header_; };
         Trailer trailer() const { return trailer_; };

         inline unsigned int blocks() const { return block_header_.getBlocks(); };
         // Returns the size of the payload _without_ the headers
         inline unsigned int size() const { return payload_.size() - 3; };

      private:
         BlockHeader block_header_;
         Header header_;
         Trailer trailer_;

         std::vector<uint64_t> payload_;
   };
}

#endif
