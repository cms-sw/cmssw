//      ====================================================================
//
//      Guid.cpp
//      --------------------------------------------------------------------
//
//      Package   : Persistent Guid to identify objects in the persistent
//              world.
//
//      Author    : Markus Frank
//
//      ====================================================================
#include "Guid.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdlib.h>
#include <string>
#include "uuid/uuid.h"

namespace edm {
  static char const* fmt_Guid =
    "%08lX-%04hX-%04hX-%02hhX%02hhX-%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX";

  static int const bufSize = 128;

  /// Initialize a new Guid
  void Guid::init()   {
    uuid_t me_;
    ::uuid_generate_time(me_);
    unsigned int*   d1 = reinterpret_cast<unsigned int*>(me_);
    unsigned short* d2 = reinterpret_cast<unsigned short*>(me_+4);
    unsigned short* d3 = reinterpret_cast<unsigned short*>(me_+6);
    Data1 = *d1;
    Data2 = *d2;
    Data3 = *d3;
    for(int i = 0; i < 8; ++i){
      Data4[i] = me_[i + 8];
    }
  }

  std::string const Guid::toString() const {
    char text[bufSize];
    ::snprintf(text, sizeof(text),
              fmt_Guid,
              Data1, Data2, Data3,
              Data4[0], Data4[1], Data4[2], Data4[3],
              Data4[4], Data4[5], Data4[6], Data4[7]);
    return text;
  }

  // fromString is used only in a unit test, so performance is not critical.
  Guid const& Guid::fromString(std::string const& source) {
    char const dash = '-';
    size_t const iSize = 8;
    size_t const sSize = 4;
    size_t const cSize = 2;
    size_t offset = 0;
    Data1 = strtol(source.substr(offset, iSize).c_str(), 0, 16);
    offset += iSize;
    assert(dash == source[offset++]); 
    Data2 = strtol(source.substr(offset, sSize).c_str(), 0, 16);
    offset += sSize;
    assert(dash == source[offset++]); 
    Data3 = strtol(source.substr(offset, sSize).c_str(), 0, 16);
    offset += sSize;
    assert(dash == source[offset++]); 
    Data4[0] = strtol(source.substr(offset, cSize).c_str(), 0, 16);
    offset += cSize;
    Data4[1] = strtol(source.substr(offset, cSize).c_str(), 0, 16);
    offset += cSize;
    assert(dash == source[offset++]);
    Data4[2] = strtol(source.substr(offset, cSize).c_str(), 0, 16);
    offset += cSize;
    Data4[3] = strtol(source.substr(offset, cSize).c_str(), 0, 16);
    offset += cSize;
    Data4[4] = strtol(source.substr(offset, cSize).c_str(), 0, 16);
    offset += cSize;
    Data4[5] = strtol(source.substr(offset, cSize).c_str(), 0, 16);
    offset += cSize;
    Data4[6] = strtol(source.substr(offset, cSize).c_str(), 0, 16);
    offset += cSize;
    Data4[7] = strtol(source.substr(offset, cSize).c_str(), 0, 16);
    offset += cSize;
    assert(source.size() == offset);
    return *this;
  }

  bool Guid::operator<(Guid const& g) const {
    return ::memcmp(&g.Data1, &Data1, 16) < 0;
  }
}
