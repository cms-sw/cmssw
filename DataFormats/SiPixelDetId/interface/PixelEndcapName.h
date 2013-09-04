#ifndef SiPixelDetId_PixelEndcapName_H
#define SiPixelDetId_PixelEndcapName_H

/** \class PixelEndcapName
 * Endcap Module name (as in PixelDatabase) for endcaps
 */

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"

#include <string>
#include <iostream>
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

class DetId;

class PixelEndcapName : public PixelModuleName {
public:

  enum HalfCylinder { mO = 1, mI = 2 , pO =3 , pI =4 };

  /// ctor from DetId
  PixelEndcapName(const DetId &);
  
  /// ctor for defined name
  PixelEndcapName( HalfCylinder part = mO, int disk =0, int blade =0, int pannel=0, int plaq=0) 
    : PixelModuleName(false), 
      thePart(part), theDisk(disk), theBlade(blade), thePannel(pannel), thePlaquette(plaq)
  { }

  /// ctor from name string
  PixelEndcapName(std::string name);

  virtual ~PixelEndcapName() { }

  /// from base class
  virtual std::string name() const;

  HalfCylinder halfCylinder() const { return thePart; } 

  /// disk id
  uint32_t diskName() const { return theDisk; }

  /// blade id
  uint32_t bladeName() const { return theBlade; }

  /// pannel id 
  uint32_t pannelName() const { return thePannel; }

  /// plaquetteId (in pannel)
  uint32_t plaquetteName() const { return thePlaquette; }

  /// module Type
   virtual PixelModuleName::ModuleType  moduleType() const;

  /// return DetId
  PXFDetId getDetId(); 

  /// check equality of modules from datamemebers
  virtual bool operator== (const PixelModuleName &) const;


private:
  HalfCylinder thePart;
  uint32_t theDisk, theBlade, thePannel, thePlaquette;
};

std::ostream & operator<<( std::ostream& out, const PixelEndcapName::HalfCylinder & t);
#endif
