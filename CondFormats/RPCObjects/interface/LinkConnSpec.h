#ifndef CondFormatsRPCObjectsLinkConnSpec_H
#define CondFormatsRPCObjectsLinkConnSpec_H

#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include <vector>

/** \class LinkConnSpec
 * RPC LinkConnection Specification for readout decoding
 */

class LinkConnSpec {
public:
  
  /// ctor with ID only
  LinkConnSpec(int num=-1) : theTriggerBoardInputNumber(num) { }

  /// this link input number in TriggerBoard
  int triggerBoardInputNumber() const { return theTriggerBoardInputNumber; }

  /// LB served by this link, identified by its position in link 
  const LinkBoardSpec * linkBoard(int linkBoardNumInLink) const;
  
  /// attach LinkBoard to this link
  void add(const LinkBoardSpec & lb);

  ///  debud printaout, call its components with depth dectreased by one
  void print(int depth = 0) const;

private:
  int theTriggerBoardInputNumber;
  std::vector<LinkBoardSpec> theLBs;
};
#endif
