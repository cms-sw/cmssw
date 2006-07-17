#ifndef CondFormatsRPCObjectsFebSpec_H
#define CondFormatsRPCObjectsFebSpec_H

#include <vector>
#include <string>
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"
#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"

/** \class FebSpec 
 * RPC FEB specification for readout decoding
 */

class ChamberLocationSpec;

class FebSpec {
public:
  /// ctor with ID only
  FebSpec(int num=-1, 
      RPCReadOutMapping::ChamberLocationSepcReference chamberLocationRef = -1,
      std::string   cmsRoll = "", int posInCmsRoll = 0,
      std::string localRoll = "", int posInLocalRoll =0) 
    : theLinkBoardInputNum(num), 
      theChamberSpecRef(chamberLocationRef),
      theFebCmsEtaPartition(cmsRoll),
      theFebPositionInCmsEtaPartition(posInCmsRoll),
      theFebLocalEtaPartition(localRoll), 
      theFebPositionInLocalEtaPartition(posInLocalRoll),
      theRawId(0) { }

  /// this FEB channel in LinkBoard
  int linkBoardInputNum() const { return theLinkBoardInputNum; }

  /// add strip
  void add(const ChamberStripSpec & strip);

  /// strip info for input pin
  const ChamberStripSpec * strip(int pinNumber) const; 

  /// local FEB postion in Roll 
  int febPositionInLocalEtaPartition() const { 
      return theFebPositionInLocalEtaPartition;}

  /// local FEB roll
  std::string febLocalEtaPartition() const {
    return theFebLocalEtaPartition;
  }

  /// cms FEB postion in Roll 
  int febPositionInCmsEtaPartition() const { 
      return theFebPositionInCmsEtaPartition;}

  /// cms FEB roll
  std::string febCmsEtaPartition() const {
    return theFebCmsEtaPartition;
  }

  /// reference to chamber
  RPCReadOutMapping::ChamberLocationSepcReference 
      chamberLocationSpecReference() const { 
    return theChamberSpecRef;
  } 

  /// debug printout
  void print(int depth) const;

private:

  int theLinkBoardInputNum;
  RPCReadOutMapping::ChamberLocationSepcReference theChamberSpecRef; 
  std::string theFebCmsEtaPartition;
  int theFebPositionInCmsEtaPartition;
  std::string theFebLocalEtaPartition;
  int theFebPositionInLocalEtaPartition;
   
  std::vector<ChamberStripSpec> theStrips;
  mutable uint32_t theRawId;
};
#endif
