#ifndef L1Trigger_CSCTriggerPrimitives_PulseArray_h
#define L1Trigger_CSCTriggerPrimitives_PulseArray_h

/** \class PulseArray
 *
 */

#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCPatternBank.h"

class PulseArray {
public:
  // constructor
  PulseArray();

  // set the dimensions
  void initialize(unsigned numberOfChannels);

  // clear the pulse array
  void clear();

  unsigned& operator()(const unsigned layer, const unsigned channel);

  unsigned bitsInPulse() const;

  // make the pulse at time "bx" with length "hit_persist"
  void extend(const unsigned layer, const unsigned channel, const unsigned bx, const unsigned hit_persist);

  // check "one shot" at this bx_time
  bool oneShotAtBX(const unsigned layer, const unsigned channel, const unsigned bx) const;

  // check if "one shot" is high at this bx_time
  bool isOneShotHighAtBX(const unsigned layer, const unsigned channel, const unsigned bx) const;

  // This loop is a quick check of a number of layers hit at bx_time: since
  // most of the time it is 0, this check helps to speed-up the execution
  // substantially.
  unsigned numberOfLayersAtBX(const unsigned bx) const;

private:
  std::vector<std::vector<unsigned> > data_;
  unsigned numberOfChannels_;
};

#endif
