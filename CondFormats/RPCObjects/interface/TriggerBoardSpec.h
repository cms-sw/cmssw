#ifndef CondFormatsRPCObjectsTriggerBoardSpec_H
#define CondFormatsRPCObjectsTriggerBoardSpec_H

#include <boost/cstdint.hpp>
#include "CondFormats/RPCObjects/interface/LinkConnSpec.h"

/** \class TriggerBoardSpec
 * RPC Trigger Board specification for readout decoding
 */

class TriggerBoardSpec {
public:
  /// ctor with ID only
  TriggerBoardSpec(int num=-1);

  /// input channel number to DCC 
  int dccInputChannelNum() const { return theNum; }

  /// link attached to this TB with given input number
  const LinkConnSpec * linkConn(int tbInputNumber) const;

  ///  attach connection to TB
  void add(const LinkConnSpec & lc);

  ///  debud printaout, call its components with depth dectreased by one
  void print(int depth = 0) const;
  
private:
  int theNum;
  std::vector<LinkConnSpec> theLinks;
};
#endif
