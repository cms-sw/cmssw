#ifndef DQMOFFLINE_LUMI_TRIGGERDEFS_H
#define DQMOFFLINE_LUMI_TRIGGERDEFS_H

#include <bitset>

const unsigned int kNTrigBit = 128;
typedef std::bitset<kNTrigBit> TriggerBits;
const unsigned int kNTrigObjectBit = 256;
typedef std::bitset<kNTrigObjectBit> TriggerObjects;

#endif
