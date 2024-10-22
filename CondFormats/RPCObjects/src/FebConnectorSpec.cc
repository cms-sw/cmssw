#include "CondFormats/RPCObjects/interface/FebConnectorSpec.h"
#include "CondFormats/RPCObjects/interface/DBSpecToDetUnit.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <sstream>

FebConnectorSpec::FebConnectorSpec(int num, const ChamberLocationSpec& chamber, const FebLocationSpec& feb)
    : theLinkBoardInputNum(num), theChamber(chamber), theFeb(feb), theAlgo(0), theRawId(0) {}

FebConnectorSpec::FebConnectorSpec(FebConnectorSpec const& iOther)
    : theLinkBoardInputNum(iOther.theLinkBoardInputNum),
      theChamber(iOther.theChamber),
      theFeb(iOther.theFeb),
      theAlgo(iOther.theAlgo),
      theRawId(iOther.theRawId.load()) {}

FebConnectorSpec& FebConnectorSpec::operator=(FebConnectorSpec const& iOther) {
  theLinkBoardInputNum = iOther.theLinkBoardInputNum;
  theChamber = iOther.theChamber;
  theFeb = iOther.theFeb;
  theAlgo = iOther.theAlgo;
  theRawId.store(iOther.theRawId.load());
  return *this;
}

const ChamberStripSpec FebConnectorSpec::strip(int pinNumber) const {
  int nStrips = theAlgo / 10000;
  int firstChamberStrip = (theAlgo - 10000 * nStrips) / 100;
  int pinAlgo = theAlgo - 10000 * nStrips - 100 * firstChamberStrip;
  int slope = 1;
  if (pinAlgo > 3) {
    pinAlgo = pinAlgo - 4;
    slope = -1;
  }
  bool valid = true;
  if (pinNumber < pinAlgo)
    valid = false;
  if (!pinAlgo && (pinNumber < 2))
    valid = false;
  if (pinAlgo && (pinNumber > pinAlgo + nStrips - 1))
    valid = false;
  if (!pinAlgo && (pinNumber > nStrips + 2 || pinNumber == 9))
    valid = false;
  int chamberStripNumber = -1;
  if (valid) {
    if (pinAlgo != 0)
      chamberStripNumber = firstChamberStrip + slope * (pinNumber - pinAlgo);
    else if (pinNumber < 9)
      chamberStripNumber = firstChamberStrip + slope * (pinNumber - 2);
    else
      chamberStripNumber = firstChamberStrip + slope * (pinNumber - 3);
  }
  ChamberStripSpec aStrip = {pinNumber, chamberStripNumber, 0};
  return aStrip;
}

const int FebConnectorSpec::chamberStripNum(int istrip) const {
  int nStrips = theAlgo / 10000;
  if (istrip < 0 || istrip > nStrips - 1)
    return 0;
  int firstChamberStrip = (theAlgo - 10000 * nStrips) / 100;
  int pinAlgo = theAlgo - 10000 * nStrips - 100 * firstChamberStrip;
  int theStrip = firstChamberStrip + istrip;
  if (pinAlgo > 3)
    theStrip = firstChamberStrip - istrip;
  return theStrip;
}

const int FebConnectorSpec::cablePinNum(int istrip) const {
  int nStrips = theAlgo / 10000;
  if (istrip < 0 || istrip > nStrips - 1)
    return 0;
  int pinAlgo = theAlgo % 100;
  if (pinAlgo > 3)
    pinAlgo = pinAlgo - 4;
  bool holeatpin9 = (pinAlgo == 0 && istrip > 6);
  int thePin = istrip + pinAlgo + holeatpin9 + 2 * (pinAlgo == 0);
  return thePin;
}

uint32_t FebConnectorSpec::rawId() const {
  DBSpecToDetUnit toDU;
  if (!theRawId) {
    uint32_t expected = 0;
    theRawId.compare_exchange_strong(expected, toDU(theChamber, theFeb));
  }
  return theRawId.load();
}

std::string FebConnectorSpec::print(int depth) const {
  std::ostringstream str;
  str << "FebConnectorSpec in LinkBoardNum =" << linkBoardInputNum() << " rawId: " << rawId() << std::endl;
  RPCDetId aDet(rawId());
  str << aDet << std::endl;
  str << theChamber.print(depth) << std::endl << theFeb.print(depth);
  depth--;
  if (depth >= 0) {
    int nStrips = theAlgo / 10000;
    for (int istrip = 0; istrip < nStrips; istrip++) {
      ChamberStripSpec aStrip = {cablePinNum(istrip), chamberStripNum(istrip), cmsStripNum(istrip)};
      str << aStrip.print(depth);
    }
  }
  return str.str();
}
