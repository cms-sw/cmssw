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
#include <cstdio>
#include <cstring>
#include "uuid/uuid.h"

namespace edm {
  static char const* fmt_Guid = 
    "%08lX-%04hX-%04hX-%02hhX%02hhX-%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX";
  
  //{ 0x0,0x0,0x0,{0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0}};
  static Guid const
    clid_null(std::string("00000000-0000-0000-0000-000000000000"));
  
  Guid const& Guid::null() {
    return clid_null;
  }
  
  /// Create a new Guid
  void Guid::create(Guid& guid)   {
    uuid_t me_;
    ::uuid_generate_time(me_);
    unsigned int   *d1 = reinterpret_cast<unsigned int*>(me_);
    unsigned short *d2 = reinterpret_cast<unsigned short*>(me_+4);
    unsigned short *d3 = reinterpret_cast<unsigned short*>(me_+6);
    guid.Data1 = *d1;
    guid.Data2 = *d2;
    guid.Data3 = *d3;
    for (int i = 0; i < 8; ++i){
      guid.Data4[i]=me_[i+8];
    }
  }
  
  std::string const Guid::toString() const {
    char text[128];
    ::sprintf(text, fmt_Guid,
              Data1, Data2, Data3, 
              Data4[0], Data4[1], Data4[2], Data4[3], 
              Data4[4], Data4[5], Data4[6], Data4[7]);
    return text;
  }
  
  Guid const& Guid::fromString(std::string const& source) {
    // Note: This looks funny, but the specs for sscanf formats say
    //       that the space of a pointer in the ellipsis may only be
    //       integer or short. Hence one has to reserve a bit more space
    //       otherwise the stack gets corrupted.
    unsigned char d[8];
    ::sscanf( source.c_str(), fmt_Guid, &Data1, &Data2, &Data3, 
              &Data4[0], &Data4[1], &Data4[2], &Data4[3], &d[0], &d[1], &d[2], &d[3]);
    //*(int*)&Data4[4] = *(int*)d;
    unsigned int      * p = reinterpret_cast<unsigned int *>(&Data4[4]);
    unsigned int const* q = reinterpret_cast<unsigned int const*>(&d[0]);
    *p = *q;
    return *this;
  }
  
  bool Guid::operator<(Guid const& g) const {
    return ::memcmp(&g.Data1, &Data1, 16) < 0;
  }
}
