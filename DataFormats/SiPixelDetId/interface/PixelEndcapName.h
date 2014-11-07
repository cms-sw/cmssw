#ifndef SiPixelDetId_PixelEndcapName_H
#define SiPixelDetId_PixelEndcapName_H

/** \class PixelEndcapName
 * Endcap Module name (as in PixelDatabase) for endcaps
 */
#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameBase.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

class DetId;

class PixelEndcapName : public PixelModuleName, public PixelEndcapNameBase {
public:

  /// ctor from DetId
  PixelEndcapName(const DetId &, bool phase=false);
  PixelEndcapName(const DetId &, const TrackerTopology* tt, bool phase=false);
  
  /// ctor for defined name
  PixelEndcapName( PixelEndcapNameBase::HalfCylinder part = mO, int disk =0, int blade =0, int pannel=0, 
		   int plaq=0, bool phase=false) 
    : PixelModuleName(false), 
      PixelEndcapNameBase(part, disk, blade, pannel, plaq), 
    phase1(phase)
  { }

  /// ctor from name string
  PixelEndcapName(std::string name);

  virtual ~PixelEndcapName() { }

  /// from base class
  virtual std::string name() const;

  /// module Type
   virtual PixelModuleName::ModuleType  moduleType() const;

  /// return DetId
  PXFDetId getDetId(); 
  DetId getDetId(const TrackerTopology* tt); 

  /// check equality of modules from datamemebers
  virtual bool operator== (const PixelModuleName &) const;


private:
  bool phase1;
};

#endif
