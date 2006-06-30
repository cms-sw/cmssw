#ifndef CondFormatsRPCObjectsFebSpec_H
#define CondFormatsRPCObjectsFebSpec_H

#include <vector>
#include <string>
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"

/** \class FebSpec 
 * RPC FEB specification for readout decoding
 */

class ChamberLocationSpec;

class FebSpec {
public:
  /// ctor with ID only
  FebSpec(int num=-1, std::string roll = "", int posInRoll =0) 
    : theLinkBoardInputNum(num), 
      theFebLocalEtaPartition(roll), 
      theFebPositionInLocalEtaPartition(posInRoll),
      theRawId(0) { }

  /// this FEB channel in LinkBoard
  int linkBoardInputNum() const { return theLinkBoardInputNum; }

  /// add strip
  void add(const ChamberStripSpec & strip);

  /// strip info for input pin
  const ChamberStripSpec * strip(int pinNumber) const; 

  /// FEB postion in Roll 
  int febPositionInLocalEtaPartition() const { 
      return theFebPositionInLocalEtaPartition;}

  /// FEB roll
  std::string febLocalEtaPartition() const {
    return theFebLocalEtaPartition;
  }

  /// DetUnit in which Fed belongs to 
  const uint32_t & rawId(const ChamberLocationSpec & location) const;  

  /// debug printout
  void print(int depth) const;

private:

  int theLinkBoardInputNum;
  std::string theFebLocalEtaPartition;
  int theFebPositionInLocalEtaPartition;
   
  std::vector<ChamberStripSpec> theStrips;
  mutable uint32_t theRawId;
};
#endif
