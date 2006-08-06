#ifndef CondFormatsRPCObjectsFebConnectorSpec_H
#define CondFormatsRPCObjectsFebConnectorSpec_H

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"

/** \class FebConnectorSpec 
 * Specifies the input for LinkBoard. In hardware the data goes through
 * FEB connector which collects data from input strips.
 * This class provides access to strip on one side and DetUnit location
 * (through ChamberLocationSpec and FebSpec info) on the other side.
 *
 * FIXME - afer debug fill theRawId in constructor and remove theChamber,theFeb
 *
 */

class FebConnectorSpec {
public:
  FebConnectorSpec(int num =-1) : theLinkBoardInputNum(num), theRawId(0) { }
  FebConnectorSpec(int num, const ChamberLocationSpec & chamber, const FebLocationSpec & feb); 

  /// this FEB channel in LinkBoard
  int linkBoardInputNum() const { return theLinkBoardInputNum; }

  /// add strip
  void add(const ChamberStripSpec & strip);

  /// strip info for input pin
  const ChamberStripSpec * strip(int pinNumber) const;

  /// DetUnit to which data belongs
  const uint32_t & rawId() const;

  const ChamberLocationSpec & chamber() const { return theChamber; }
  const FebLocationSpec     & feb()  const { return theFeb; }

  /// debug
  void print(int depth=0) const;

private:
  int theLinkBoardInputNum;

  ChamberLocationSpec theChamber; 
  FebLocationSpec     theFeb; 

  std::vector<ChamberStripSpec> theStrips;
  mutable uint32_t theRawId;
};

#endif
