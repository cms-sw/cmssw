#include "CondCore/ORA/interface/Guid.h"
// externals
#include "uuid/uuid.h"
//
#include <cstring>
#include <cstdio>

static const char* fmt_Guid = "%08lX-%04hX-%04hX-%02hhX%02hhX-%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX";
static const char* guid_null = "00000000-0000-0000-0000-000000000000";

std::string ora::Guid::null(){
  return guid_null;
}

void ora::Guid::fromTime() {
  uuid_t me_;
  ::uuid_generate_time(me_);
  unsigned int*  tmp = reinterpret_cast<unsigned int*>(me_);
  Data1 = *tmp;
  Data2 = *reinterpret_cast<unsigned short*>(me_+4);
  Data3 = *reinterpret_cast<unsigned short*>(me_+6);
  for (int i = 0; i < 8; ++i){
    Data4[i]=me_[i+8];
  }
}

std::string ora::Guid::toString() const {

  char text[GUID_STRING_SIZE];
  ::snprintf(text, GUID_STRING_SIZE, fmt_Guid,
	     Data1, Data2, Data3,
	     Data4[0], Data4[1], Data4[2], Data4[3],
	     Data4[4], Data4[5], Data4[6], Data4[7]);
  return text;
}

std::string ora::guidFromTime() {
  Guid tmp;
  tmp.fromTime();
  return tmp.toString();
}

  
