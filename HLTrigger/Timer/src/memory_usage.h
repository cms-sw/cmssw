#ifndef memory_usage_h
#define memory_usage_h

#include <cstdint>

class memory_usage {
public:
  static bool     is_available();
  static uint64_t allocated();
  static uint64_t deallocated();
};

#endif // memory_usage_h
