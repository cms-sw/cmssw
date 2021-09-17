#ifndef DataFormats_TrackerCommon_PixelEndcapName_H
#define DataFormats_TrackerCommon_PixelEndcapName_H

/** \class PixelEndcapName
 * Endcap Module name (as in PixelDatabase) for endcaps
 */
#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include <string>
#include <iostream>

class DetId;
class TrackerTopology;

class PixelEndcapName : public PixelModuleName {
public:
  enum HalfCylinder { mO = 1, mI = 2, pO = 3, pI = 4 };

  /// ctor from DetId
  PixelEndcapName(const DetId&, bool phase = false);
  PixelEndcapName(const DetId&, const TrackerTopology* tt, bool phase = false);

  /// ctor for defined name
  PixelEndcapName(HalfCylinder part = mO, int disk = 0, int blade = 0, int pannel = 0, int plaq = 0, bool phase = false)
      : PixelModuleName(false),
        thePart(part),
        theDisk(disk),
        theBlade(blade),
        thePannel(pannel),
        thePlaquette(plaq),
        phase1(phase) {}

  /// ctor from name string
  PixelEndcapName(std::string name, bool phase = false);

  ~PixelEndcapName() override {}

  /// from base class
  std::string name() const override;

  HalfCylinder halfCylinder() const { return thePart; }

  /// disk id
  int diskName() const { return theDisk; }

  /// blade id
  int bladeName() const { return theBlade; }

  /// pannel id
  int pannelName() const { return thePannel; }

  /// plaquetteId (in pannel)
  int plaquetteName() const { return thePlaquette; }

  /// ring Id
  int ringName() const { return thePlaquette; }

  /// module Type
  PixelModuleName::ModuleType moduleType() const override;

  /// return DetId
  PXFDetId getDetId();
  DetId getDetId(const TrackerTopology* tt);

  /// check equality of modules from datamemebers
  bool operator==(const PixelModuleName&) const override;

private:
  HalfCylinder thePart;
  int theDisk, theBlade, thePannel, thePlaquette;
  bool phase1;
};

std::ostream& operator<<(std::ostream& out, const PixelEndcapName::HalfCylinder& t);
#endif
