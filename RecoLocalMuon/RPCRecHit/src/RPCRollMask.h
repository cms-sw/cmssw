#ifndef RecoLocalMuon_RPCRollMask_h
#define RecoLocalMuon_RPCRollMask_h

#include <bitset>
#include <vector>

const int maskSIZE=260; // in principle 192 would be enough, for now made compatible also with 256 strips but this section should be reprogrammed
typedef std::bitset<maskSIZE> RollMask;

#endif
