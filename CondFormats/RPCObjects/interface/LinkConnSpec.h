#ifndef CondFormatsRPCObjectsLinkConnSpec_H
#define CondFormatsRPCObjectsLinkConnSpec_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include <vector>
#include <string>

/** \class LinkConnSpec
 * RPC LinkConnection Specification for readout decoding
 */

class LinkConnSpec {
public:
  /// ctor with ID only
  LinkConnSpec(int num = -1) : theTriggerBoardInputNumber(num) {}

  /// this link input number in TriggerBoard
  int triggerBoardInputNumber() const { return theTriggerBoardInputNumber; }

  /// LB served by this link, identified by its position in link
  const LinkBoardSpec* linkBoard(int linkBoardNumInLink) const;

  const std::vector<LinkBoardSpec>& linkBoards() const { return theLBs; }

  /// attach LinkBoard to this link
  void add(const LinkBoardSpec& lb);

  ///  debud printaout, call its components with depth dectreased by one
  std::string print(int depth = 0) const;

private:
  int theTriggerBoardInputNumber;
  std::vector<LinkBoardSpec> theLBs;

  COND_SERIALIZABLE;
};
#endif
