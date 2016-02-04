#include "FWCore/Utilities/interface/Adler32Calculator.h"

namespace cms {

  //-------------------------------------------------------
  // the following is adapted from 
  // http://en.wikipedia.org/wiki/Adler-32
  //-------------------------------------------------------
  
  void
  Adler32(char const* data, size_t len, uint32_t& a, uint32_t& b) {
   /* data: Pointer to the data to be summed; len is in bytes */
  
    #define MOD_ADLER 65521
   
    unsigned char const* ptr = static_cast<unsigned char const*>(static_cast<void const*>(data));
    while (len > 0) {
      size_t tlen = (len > 5552 ? 5552 : len);
      len -= tlen;
      do {
        a += *ptr++;
        b += a;
      } while (--tlen);
      
      a %= MOD_ADLER;
      b %= MOD_ADLER;
    }
  
    #undef MOD_ADLER
  }
  
  uint32_t
  Adler32(char const* data, size_t len) {
   /* data: Pointer to the data to be summed; len is in bytes */
    uint32_t a = 1, b = 0;
    Adler32(data, len, a, b);
    return (b << 16) | a;
  }
}
