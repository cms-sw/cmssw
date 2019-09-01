#ifndef CondFormatsRPCObjectsLinkBoardSpec_H
#define CondFormatsRPCObjectsLinkBoardSpec_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/RPCObjects/interface/FebConnectorSpec.h"
#include <string>

/** \class LinkBoardSpec
 * RPC LinkBoard Specification for readout decoding. Provide chamber location specification (as in DB throught FEBs) 
 */

class LinkBoardSpec {
public:
  /// dummy
  LinkBoardSpec() : theMaster(false) {}

  /// real ctor specifyig LB if this LB is master,
  /// its number in link, and which chamber it is serving
  LinkBoardSpec(bool master, int linkBoardNumInLin, int lbCode);

  /// true if master LB (is it of any use?)
  bool master() { return theMaster; }

  /// this LB number in link
  int linkBoardNumInLink() const { return theLinkBoardNumInLink; }

  /// LB name as in OMDS
  std::string linkBoardName() const;

  /// attach feb
  void add(const FebConnectorSpec& feb);

  /// get Feb by its connection number to this board
  const FebConnectorSpec* feb(int febInputNum) const;
  const std::vector<FebConnectorSpec>& febs() const { return theFebs; }

  /// debud printout
  std::string print(int depth = 0) const;

private:
  bool theMaster;
  int theLinkBoardNumInLink;
  int theCode;
  std::vector<FebConnectorSpec> theFebs;

  COND_SERIALIZABLE;
};
#endif
