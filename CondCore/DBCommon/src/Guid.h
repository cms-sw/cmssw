#ifndef CondCore_DBCommon_Guid_h
#define CondCore_DBCommon_Guid_h

#include <cstring>
#include <cstdio>

static const char* fmt_Guid = "%08lX-%04hX-%04hX-%02hhX%02hhX-%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX";

struct Guid {
  unsigned int  Data1;
  unsigned short Data2;
  unsigned short Data3;
  unsigned char  Data4[8];
  std::string toString() const {
    char text[128];
    ::sprintf(text, fmt_Guid,
	      Data1, Data2, Data3,
	      Data4[0], Data4[1], Data4[2], Data4[3],
	      Data4[4], Data4[5], Data4[6], Data4[7]);
    return text;
  }
};

#endif //CondCore_DBCommon_Guid_h
