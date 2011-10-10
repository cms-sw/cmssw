#ifndef INCLUDE_COND_GUID_H
#define INCLUDE_COND_GUID_H

#define GUID_STRING_SIZE 40

#include <string>

namespace ora {

  struct Guid {
    static std::string null();
    unsigned int  Data1;
    unsigned short Data2;
    unsigned short Data3;
    unsigned char  Data4[8];
    void fromTime();
    std::string toString() const;
  };

  std::string guidFromTime();

}

namespace cond {

  typedef ora::Guid Guid;
  void* genMD5(void* buffer, unsigned long len, void* code);
  void genMD5(const std::string& s, void* code);

}


#endif 
