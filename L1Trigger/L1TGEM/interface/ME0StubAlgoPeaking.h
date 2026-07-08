#ifndef L1Trigger_L1TGEM_ME0StubAlgoPeaking_H
#define L1Trigger_L1TGEM_ME0StubAlgoPeaking_H

#include <vector>
#include <algorithm>
#include "L1Trigger/L1TGEM/interface/ME0StubPrimitive.h"

namespace l1t {
  namespace me0 {

    class PeakingManager {
    private:
      std::vector<std::vector<std::vector<ME0StubPrimitive>>>
          segs_;                                // [oldest / old][partition][sbit] : size = (2, 15, 192)
      std::vector<std::vector<bool>> trigger_;  // [partition][sbit]
    public:
      PeakingManager() = default;
      PeakingManager(const int numPart = 15, const int width = 192) {
        segs_ = std::vector<std::vector<std::vector<ME0StubPrimitive>>>(
            2, std::vector<std::vector<ME0StubPrimitive>>(numPart, std::vector<ME0StubPrimitive>(width)));
        trigger_ = std::vector<std::vector<bool>>(numPart, std::vector<bool>(width, false));
      }
      std::vector<ME0StubPrimitive> processSegments(const int partition, const std::vector<ME0StubPrimitive>& newSegs);
      std::vector<std::vector<bool>> getTrigger() const { return trigger_; };
      std::vector<std::vector<std::vector<ME0StubPrimitive>>> getSegs() const { return segs_; };
    };
  }  // namespace me0
}  // namespace l1t
#endif

/*
peaking algorithm
--> trigger is only set when (oldest, old, new) = (not exist, exist, exist), and privous trigger is not set.
--> output is old segment when trigged, or (oldest, old, new) = (not exist, exist, not exist). Otherwise, output is empty segment.
ex)
   BX | oldest | old | new | trigger | output
   ---------------------------------------------------------------
   0  |   0    |  0  |  1  |    0    | empty
   1  |   0    |  1  |  0  |    0    | old (not trigged as new segment is not existed)
   2  |   1    |  0  |  0  |    0    | empty (not trigged as old segment is not existed)
   3  |   0    |  0  |  0  |    0    | empty
(possible case : a same segment exists for more than 2 BXs, but only trigger once when it is existed for the first time while oldest segment is not existed.)
   BX | oldest | old | new | trigger | output
   ---------------------------------------------------------------
   0  |   0    |  0  |  1  |    0    | empty
   1  |   0    |  1  |  1  |    0    | empty
   2  |   1    |  1  |  1  |    1    | old (trigged from BX 1)
   3  |   1    |  1  |  1  |    0    | empty (trigger is reset after firing at BX 2, so not trigged at BX 3)
   4  |   1    |  1  |  1  |    0    | empty (trigger is not set when 3 consecutive segments exist, so not trigged at BX 4)
   5  |   1    |  1  |  0  |    0    | empty
   6  |   1    |  0  |  0  |    0    | empty
*/