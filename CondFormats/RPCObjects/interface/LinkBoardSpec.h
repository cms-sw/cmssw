#ifndef CondFormatsRPCObjectsLinkBoardSpec_H
#define CondFormatsRPCObjectsLinkBoardSpec_H

#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"

/** \class LinkBoardSpec
 * RPC LinkBoard Specification for readout decoding. Provide chamber location specification (as in DB throught FEBs) 
 */

class LinkBoardSpec {
public:
  /// dummy
  LinkBoardSpec() {}

  /// real ctor specifyig LB if this LB is master, 
  /// its number in link, and which chamber it is serving  
  LinkBoardSpec(bool master, int linkBoardNumInLin, const ChamberLocationSpec & c);

  /// true if master LB (is it of any use?)
  bool master() { return theMaster; }

  /// this LB number in link
  int linkBoardNumInLink() const { return theLinkBoardNumInLink; }

  /// serving chamber location info
  const ChamberLocationSpec & chamberLocationSpec() const 
      { return theChamberSpec; }

  /// debud printout
  void print(int depth=0) const;

private: 
  bool theMaster;
  int theLinkBoardNumInLink; 
  ChamberLocationSpec theChamberSpec;
};
#endif
