#ifndef DataFormats_TrackerCommon_PixelBarrelName_H
#define DataFormats_TrackerCommon_PixelBarrelName_H

/** \class PixelBarrelName
 * Module name (as in PixelDatabase) in barrel
 */

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

#include <string>

class DetId;
class TrackerTopology;

class PixelBarrelName : public PixelModuleName {
public:
  enum Shell { mO = 1, mI = 2, pO = 3, pI = 4 };

  /// ctor from DetId
  PixelBarrelName(const DetId&, const TrackerTopology* tt, bool phase = false);

  // do not use, works only for phase0 and old pixel classes
  PixelBarrelName(const DetId&, bool phase = false);

  /// ctor for defined name with dummy parameters
  PixelBarrelName(Shell shell = mO, int layer = 0, int module = 0, int ladder = 0, bool phase = false)
      : PixelModuleName(true), thePart(shell), theLayer(layer), theModule(module), theLadder(ladder), phase1(phase) {}

  /// ctor from name string
  PixelBarrelName(std::string name, bool phase = false);

  ~PixelBarrelName() override {}

  inline int convertLadderNumber(int oldLadder);

  /// from base class
  std::string name() const override;

  Shell shell() const { return thePart; }

  /// layer id
  int layerName() const { return theLayer; }

  /// module id (index in z)
  int moduleName() const { return theModule; }

  /// ladder id (index in phi)
  int ladderName() const { return theLadder; }

  /// sector id
  int sectorName() const;

  /// full or half module
  bool isHalfModule() const;

  /// module Type
  PixelModuleName::ModuleType moduleType() const override;

  /// return the DetId
  PXBDetId getDetId();
  DetId getDetId(const TrackerTopology* tt);

  /// check equality of modules from datamemebers
  bool operator==(const PixelModuleName&) const override;

private:
  Shell thePart;
  int theLayer, theModule, theLadder;
  bool phase1;
};

std::ostream& operator<<(std::ostream& out, const PixelBarrelName::Shell& t);
#endif
