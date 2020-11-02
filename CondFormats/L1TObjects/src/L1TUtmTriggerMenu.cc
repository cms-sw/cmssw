// auto-generated file by import_utm.pl
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"

//copied from tmUtil
const unsigned long L1TUtmTriggerMenu::getFirmwareUuidHashed() const {
  const void* key = getFirmwareUuid().c_str();
  int len = getFirmwareUuid().size();
  unsigned int seed = 3735927486;
  return murmurHashNeutral2(key, len, seed);
}

//copied from tmUtil
unsigned long L1TUtmTriggerMenu::murmurHashNeutral2(const void* key, int len, unsigned int seed) {
  // 'm' and 'r' are mixing constants generated offline.
  // They're not really 'magic', they just happen to work well.

  const unsigned int m = 0x5bd1e995;
  const int r = 24;

  // Initialize the hash to a 'random' value

  unsigned int h = seed ^ len;

  // Mix 4 bytes at a time into the hash

  const unsigned char* data = (const unsigned char*)key;

  while (len >= 4) {
    unsigned int k;

    k = data[0];
    k |= data[1] << 8;
    k |= data[2] << 16;
    k |= data[3] << 24;

    k *= m;
    k ^= k >> r;
    k *= m;

    h *= m;
    h ^= k;

    data += 4;
    len -= 4;
  }

  // Handle the last few bytes of the input array

  switch (len) {
    case 3:
      h ^= data[2] << 16;
      [[fallthrough]];
    case 2:
      h ^= data[1] << 8;
      [[fallthrough]];
    case 1:
      h ^= data[0];
      h *= m;
  };

  // Do a few final mixes of the hash to ensure the last few
  // bytes are well-incorporated.

  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;

  return h;
}
