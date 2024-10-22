#ifndef CondFormatsRPCObjectsTriggerBoardSpec_H
#define CondFormatsRPCObjectsTriggerBoardSpec_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/RPCObjects/interface/LinkConnSpec.h"
#include <string>
#include <cstdint>

/** \class TriggerBoardSpec
 * RPC Trigger Board specification for readout decoding
 */

class TriggerBoardSpec {
public:
  /// ctor with ID only
  TriggerBoardSpec(int num = -1, uint32_t aMask = 0);

  /// input channel number to DCC
  int dccInputChannelNum() const { return theNum; }

  /// link attached to this TB with given input number
  const LinkConnSpec* linkConn(int tbInputNumber) const;

  /// not masked links belonging to this TB
  std::vector<const LinkConnSpec*> enabledLinkConns() const;

  /// all links kept by this TB
  const std::vector<LinkConnSpec> linkConns() const { return theLinks; }

  ///  attach connection to TB
  void add(const LinkConnSpec& lc);

  /// set mask links
  void setMaskedLinks(uint32_t aMask) { theMaskedLinks = aMask; }

  ///  debud printaout, call its components with depth dectreased by one
  std::string print(int depth = 0) const;

private:
  int theNum;
  uint32_t theMaskedLinks;
  std::vector<LinkConnSpec> theLinks;

  COND_SERIALIZABLE;
};
#endif
