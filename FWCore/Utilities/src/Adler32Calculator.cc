#include "FWCore/Utilities/interface/Adler32Calculator.h"

#include <zlib.h>

namespace cms {

  //-------------------------------------------------------
  // Adler-32 checksum (http://en.wikipedia.org/wiki/Adler-32),
  // computed with the optimised implementation from zlib
  //-------------------------------------------------------

  void Adler32(char const* data, size_t len, uint32_t& a, uint32_t& b) {
    /* data: Pointer to the data to be summed; len is in bytes */
    constexpr uint32_t mod_adler = 65521;

    uLong adler = ((b % mod_adler) << 16) | (a % mod_adler);
    adler = adler32_z(adler, reinterpret_cast<Bytef const*>(data), len);
    a = adler & 0xffff;
    b = (adler >> 16) & 0xffff;
  }

  uint32_t Adler32(char const* data, size_t len) {
    /* data: Pointer to the data to be summed; len is in bytes */
    return adler32_z(1L, reinterpret_cast<Bytef const*>(data), len);
  }
}  // namespace cms
