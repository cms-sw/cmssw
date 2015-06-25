#ifndef AMC13_SPEC__h
#define AMC13_SPEC__h

#include <memory>
#include <vector>
#include <stdint.h>

#include "AMCSpec.h"

namespace edm {
   class Event;
}

namespace amc13 {
   class Header {
      public:
         Header() : data_(0) {};
         Header(const uint64_t *data) : data_(data[0]) {};
         Header(unsigned int namc, unsigned int orbit);

         inline uint64_t raw() const { return data_; };
         bool check() const;

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
         Trailer(unsigned int blk, unsigned int lv1, unsigned int bx);

         inline unsigned int getCRC() const { return (data_ >> CRC_shift) & CRC_mask; };
         inline unsigned int getBlock() const { return (data_ >> BlkNo_shift) & BlkNo_mask; };
         inline unsigned int getLV1ID() const { return (data_ >> LV1_shift) & LV1_mask; };
         inline unsigned int getBX() const { return (data_ >> BX_shift) & BX_mask; };

         uint64_t raw() const { return data_; };
         bool check(unsigned int crc, unsigned int block, unsigned int lv1_id, unsigned int bx) const;
         static void writeCRC(const uint64_t *start, uint64_t *end);

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

         void add(unsigned int amc_no, unsigned int board, unsigned int lv1id, unsigned int orbit, unsigned int bx, const std::vector<uint64_t>& load);
         bool parse(const uint64_t *start, const uint64_t *data, unsigned int size, unsigned int lv1, unsigned int bx, bool legacy_mc=false);
         bool write(const edm::Event& ev, unsigned char * ptr, unsigned int skip, unsigned int size) const;

         inline std::vector<amc::Packet> payload() const { return payload_; };

      private:
         Header header_;
         std::vector<amc::Packet> payload_;
   };
}

#endif
