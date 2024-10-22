#ifndef CondFormatsRPCObjectsDccSpec_H
#define CondFormatsRPCObjectsDccSpec_H

/** \ class DccSpec
 * RPC DCC (==FED) specification for redout decoding
 */

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <string>
#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"

struct ChamberLocationSpec;

class DccSpec {
public:
  /// ctor with ID only
  DccSpec(int id = -1);

  /// id of FED
  int id() const { return theId; }

  /// TB attached to channel
  const TriggerBoardSpec* triggerBoard(int channelNumber) const;
  const std::vector<TriggerBoardSpec>& triggerBoards() const { return theTBs; }

  /// attach TB to DCC. The channel is defined by TB
  void add(const TriggerBoardSpec& tb);

  /// debud printaout, call its components with depth dectreased by one
  std::string print(int depth = 0) const;

private:
  int theId;
  std::vector<TriggerBoardSpec> theTBs;

  //  static const int MIN_CHANNEL_NUMBER = 1;
  //  static const int NUMBER_OF_CHANNELS = 68;

  COND_SERIALIZABLE;
};

#endif
