#ifndef SiPixelDetId_PixelBarrelName_H
#define SiPixelDetId_PixelBarrelName_H

/** \class PixelBarrelName
 * Module name (as in PixelDatabase) in barrel
 */

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <string>

#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameBase.h"

class DetId; 

class PixelBarrelName : public PixelModuleName, public PixelBarrelNameBase {
public:

  /// ctor from DetId
  PixelBarrelName(const DetId &, bool phase=false);

  PixelBarrelName(const DetId &, const TrackerTopology* tt, bool phase=false);

  /// ctor for defined name with dummy parameters
 PixelBarrelName(Shell shell=mO, int layer=0, int module=0, int ladder=0, bool phase=false)
   : PixelModuleName(true), 
     PixelBarrelNameBase(shell, layer, module, ladder, phase) 
  { }

  /// ctor from name string
  PixelBarrelName(std::string name);

  virtual ~PixelBarrelName() { }

  inline int convertLadderNumber(int oldLadder);

  /// from base class
  virtual std::string name() const;

  /// sector id
  int sectorName() const;

  /// full or half module
  bool isHalfModule() const;
  
  /// module Type
  virtual PixelModuleName::ModuleType  moduleType() const;

  /// return the DetId
  PXBDetId getDetId();
  DetId getDetId(const TrackerTopology* tt);

  /// check equality of modules from datamemebers
  virtual bool operator== (const PixelModuleName &) const;

private:
  bool phase1;
};

#endif
