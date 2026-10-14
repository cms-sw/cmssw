#ifndef FWCore_Utilities_Adler32Calculator_h
#define FWCore_Utilities_Adler32Calculator_h

#include <cstddef>
#include <cstdint>

/*
Adler-32 checksum (http://en.wikipedia.org/wiki/Adler-32), computed with zlib.
*/

namespace cms {

  // Adler-32 checksum of a new block of data
  inline constexpr uint32_t adler32_init = 1;

  // Checksum of len bytes of data; to checksum data in several pieces, pass the checksum of the
  // previous ones as adler.
  uint32_t adler32(char const* data, size_t len, uint32_t adler = adler32_init);
}  // namespace cms
#endif
