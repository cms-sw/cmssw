#ifndef CondFormatsRPCObjectsFebConnectorSpec_H
#define CondFormatsRPCObjectsFebConnectorSpec_H

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"
#include <string>

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

  /// add strip info
  void addStrips(int algo) {
    theAlgo = algo;
  }

  /// strip info for input pin
  const ChamberStripSpec strip(int pinNumber) const;

  /// DetUnit to which data belongs
  const uint32_t & rawId() const;

  const ChamberLocationSpec & chamber() const { return theChamber; }
  const FebLocationSpec     & feb()  const { return theFeb; }

  const int nstrips() const { return theAlgo/10000; }

  const int chamberStripNo(int istrip) const {
    int nStrips = theAlgo/10000;
    if (istrip<0 || istrip>nStrips-1) return 0;
    int firstChamberStrip=(theAlgo-10000*nStrips)/100;
    int pinAlgo=theAlgo-10000*nStrips-100*firstChamberStrip;
    int theStrip=firstChamberStrip+istrip;
    if (pinAlgo>3) theStrip=firstChamberStrip-istrip;
    return theStrip; 
  }

  const int cmsStripNo(int istrip) const { return 0; }

  const int cablePinNo(int istrip) const {
    int nStrips = theAlgo/10000;
    if (istrip<0 || istrip>nStrips-1) return 0;
    int pinAlgo=theAlgo%100;
    if (pinAlgo>3) pinAlgo=pinAlgo-4;
    bool holeatpin9=(pinAlgo==0 && istrip>7);
    int thePin = istrip+pinAlgo+holeatpin9;
    return thePin;
  }

  /// debug
  std::string print(int depth=0) const;

private:
  int theLinkBoardInputNum;

  ChamberLocationSpec theChamber; 
  FebLocationSpec     theFeb; 

  int theAlgo;
  mutable uint32_t theRawId;
};

#endif
