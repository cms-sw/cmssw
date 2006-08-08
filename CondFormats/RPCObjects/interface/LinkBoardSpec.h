#ifndef CondFormatsRPCObjectsLinkBoardSpec_H
#define CondFormatsRPCObjectsLinkBoardSpec_H

#include "CondFormats/RPCObjects/interface/FebConnectorSpec.h"

/** \class LinkBoardSpec
 * RPC LinkBoard Specification for readout decoding. Provide chamber location specification (as in DB throught FEBs) 
 */

class LinkBoardSpec {
public:
  /// dummy
  LinkBoardSpec() {}

  /// real ctor specifyig LB if this LB is master, 
  /// its number in link, and which chamber it is serving  
  LinkBoardSpec(bool master, int linkBoardNumInLin);

  /// true if master LB (is it of any use?)
  bool master() { return theMaster; }

  /// this LB number in link
  int linkBoardNumInLink() const { return theLinkBoardNumInLink; }

  /// attache feb
  void add(const FebConnectorSpec & feb); 

  /// get Feb by its connection number to this board
  const FebConnectorSpec * feb(int febInputNum) const;

  /// get strip info for LB channel
  std::pair<uint32_t,int> strip(int chanelLB) const;

  /// debud printout
  void print(int depth=0) const;

private: 
  bool theMaster;
  int theLinkBoardNumInLink; 
  std::vector<FebConnectorSpec> theFebs;
};
#endif
