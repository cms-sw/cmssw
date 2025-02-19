#ifndef FWCore_Utilities_Adler32Calculator_h
#define FWCore_Utilities_Adler32Calculator_h

#include <sys/types.h>
#include <stdint.h>

/*
Code to calculate a Adler32 checksum on a file.  This code is based
on code copied from the web in the public domain.
*/

namespace cms {

  void Adler32(char const* data, size_t len, uint32_t& a, uint32_t& b);
  uint32_t Adler32(char const* data, size_t len);
}
#endif
