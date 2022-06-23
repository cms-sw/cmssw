#ifndef CondFormatsRPCObjectsFebConnectorSpec_H
#define CondFormatsRPCObjectsFebConnectorSpec_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"
#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"
#include <string>
#include <atomic>

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
  FebConnectorSpec(int num = -1) : theLinkBoardInputNum(num), theRawId(0) {}
  FebConnectorSpec(int num, const ChamberLocationSpec& chamber, const FebLocationSpec& feb);
  FebConnectorSpec(FebConnectorSpec const&);

  FebConnectorSpec& operator=(FebConnectorSpec const&);

  /// this FEB channel in LinkBoard
  int linkBoardInputNum() const { return theLinkBoardInputNum; }

  /// add strip info
  void addStrips(int algo) { theAlgo = algo; }

  /// strip info for input pin
  const ChamberStripSpec strip(int pinNumber) const;

  /// DetUnit to which data belongs
  uint32_t rawId() const;

  const ChamberLocationSpec& chamber() const { return theChamber; }
  const FebLocationSpec& feb() const { return theFeb; }

  const int nstrips() const { return theAlgo / 10000; }

  const int chamberStripNum(int istrip) const;

  const int cmsStripNum(int istrip) const { return 0; }

  const int cablePinNum(int istrip) const;

  /// debug
  std::string print(int depth = 0) const;

private:
  int theLinkBoardInputNum;

  ChamberLocationSpec theChamber;
  FebLocationSpec theFeb;

  int theAlgo;
  mutable std::atomic<uint32_t> theRawId COND_TRANSIENT;

  COND_SERIALIZABLE;
};

#endif
