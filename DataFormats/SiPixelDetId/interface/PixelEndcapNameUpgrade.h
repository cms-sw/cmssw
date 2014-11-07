#ifndef SiPixelDetId_PixelEndcapNameUpgrade_H
#define SiPixelDetId_PixelEndcapNameUpgrade_H

/** \class PixelEndcapNameUpgrade
 * Endcap Module name (as in PixelDatabase) for endcaps
 */

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameBase.h"

#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

class DetId;

class PixelEndcapNameUpgrade : public PixelModuleName, public PixelEndcapNameBase {
public:

  /// ctor from DetId
  PixelEndcapNameUpgrade(const DetId &);
  
  /// ctor for defined name
  PixelEndcapNameUpgrade( PixelEndcapNameBase::HalfCylinder part = mO, int disk =0, int blade =0, int pannel=0, int plaq=0) 
    : PixelModuleName(false), 
      PixelEndcapNameBase(part, disk, blade, pannel, plaq)
  { }

  /// ctor from name string
  PixelEndcapNameUpgrade(std::string name);

  virtual ~PixelEndcapNameUpgrade() { }

  /// from base class
  virtual std::string name() const;

  /// module Type
   virtual PixelModuleName::ModuleType  moduleType() const;

  /// return DetId
  PXFDetId getDetId(); 

  /// check equality of modules from datamemebers
  virtual bool operator== (const PixelModuleName &) const;

};

#endif
