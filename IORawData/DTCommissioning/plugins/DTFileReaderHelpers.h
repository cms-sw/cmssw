#include <cstdint>
#ifndef DaqSource_DTFileReaderHelpers_h
#define DaqSource_DTFileReaderHelpers_h

template <class T>
char* dataPointer(const T* ptr) {
  union bPtr {
    const T* dataP;
    char* fileP;
  };
  union bPtr buf;
  buf.dataP = ptr;
  return buf.fileP;
}

template <class T>
T* typePointer(const char* ptr) {
  union bPtr {
    T* dataP;
    const char* fileP;
  };
  union bPtr buf;
  buf.fileP = ptr;
  return buf.dataP;
}

struct twoNibble {
  uint16_t lsBits;
  uint16_t msBits;
};

struct twoNibble64 {
  uint32_t lsBits;
  uint32_t msBits;
};

#endif
