#ifndef IOPool_Streamer_MsgTools_h
#define IOPool_Streamer_MsgTools_h

#include <vector>
#include <string>
#include <sstream>
#include <iterator>
#include "FWCore/Utilities/interface/Algorithms.h"

namespace edm::streamer {
  // could just use the c99 names here from stdint.h
  typedef unsigned char uint8;
  typedef unsigned short uint16;
  typedef unsigned int uint32;
  typedef unsigned long long uint64;
  typedef unsigned char char_uint64[sizeof(uint64)];
  typedef unsigned char char_uint32[sizeof(uint32)];
  typedef unsigned char char_uint16[sizeof(uint16)];
  typedef std::vector<std::string> Strings;

  inline uint64 convert64(char_uint64 v) {
    // first four bytes are code,  LSB first
    unsigned long long a = v[0], b = v[1], c = v[2], d = v[3];
    unsigned long long e = v[4], f = v[5], g = v[6], h = v[7];
    a |= (b << 8) | (c << 16) | (d << 24) | (e << 32) | (f << 40) | (g << 48) | (h << 56);
    return a;
  }

  inline uint32 convert32(char_uint32 v) {
    // first four bytes are code,  LSB first
    uint32 a = v[0], b = v[1], c = v[2], d = v[3];
    a |= (b << 8) | (c << 16) | (d << 24);
    return a;
  }

  inline uint16 convert16(char_uint16 v) {
    // first four bytes are code,  LSB first
    uint16 a = v[0], b = v[1];
    a |= (b << 8);
    return a;
  }

  inline void convert(uint32 i, char_uint32 v) {
    v[0] = static_cast<unsigned char>(i & 0xff);
    v[1] = static_cast<unsigned char>((i >> 8) & 0xff);
    v[2] = static_cast<unsigned char>((i >> 16) & 0xff);
    v[3] = static_cast<unsigned char>((i >> 24) & 0xff);
  }

  inline void convert(uint16 i, char_uint16 v) {
    v[0] = static_cast<unsigned char>(i & 0xff);
    v[1] = static_cast<unsigned char>((i >> 8) & 0xff);
  }

  inline void convert(uint64 li, char_uint64 v) {
    v[0] = static_cast<unsigned char>(li & 0xff);
    v[1] = static_cast<unsigned char>((li >> 8) & 0xff);
    v[2] = static_cast<unsigned char>((li >> 16) & 0xff);
    v[3] = static_cast<unsigned char>((li >> 24) & 0xff);
    v[4] = static_cast<unsigned char>((li >> 32) & 0xff);
    v[5] = static_cast<unsigned char>((li >> 40) & 0xff);
    v[6] = static_cast<unsigned char>((li >> 48) & 0xff);
    v[7] = static_cast<unsigned char>((li >> 56) & 0xff);
  }

  namespace MsgTools {

    inline uint8* fillNames(const Strings& names, uint8* pos) {
      uint32 sz = static_cast<uint32>(names.size());
      convert(sz, pos);                            // save number of strings
      uint8* len_pos = pos + sizeof(char_uint32);  // area for length
      pos = len_pos + sizeof(char_uint32);         // area for full string of names
      bool first = true;

      for (Strings::const_iterator beg = names.begin(); beg != names.end(); ++beg) {
        if (first)
          first = false;
        else
          *pos++ = ' ';
        pos = edm::copy_all(*beg, pos);
      }
      convert((uint32)(pos - len_pos - sizeof(char_uint32)), len_pos);
      return pos;
    }

    inline void getNames(uint8* from, uint32 from_len, Strings& to) {
      // not the most efficient way to do this
      std::istringstream ist(std::string(reinterpret_cast<char*>(from), from_len));
      typedef std::istream_iterator<std::string> Iter;
      std::copy(Iter(ist), Iter(), std::back_inserter(to));
    }

  }  // namespace MsgTools
}  // namespace edm::streamer
#endif
