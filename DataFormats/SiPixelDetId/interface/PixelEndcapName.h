#ifndef SiPixelDetId_PixelEndcapName_H
#define SiPixelDetId_PixelEndcapName_H

/** \class PixelEndcapName
 * Endcap Module name (as in PixelDatabase) for endcaps
 */

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"

#include <string>
#include <iostream>

class DetId;

class PixelEndcapName : public PixelModuleName {
public:

  enum HalfCylinder { mO = 1, mI = 2 , pO =3 , pI =4 };

  /// ctor from DetId
  PixelEndcapName(const DetId &);
  
/*
  /// ctor for defined name
  PixelEndcapName(int endcap, int disk, int blade, int pannel, int plaquette) 
    : PixelModuleName(false), theEndCap(endcap), theDisk(disk), 
      theBlade(blade), thePannel(pannel), thePlaquette(plaquette)
  { }
*/

  virtual ~PixelEndcapName() { }

  /// from base class
  virtual std::string name() const;

  HalfCylinder halfCylinder() const { return thePart; } 

  /// disk id
  int diskName() const { return theDisk; }

  /// blade id
  int bladeName() const { return theBlade; }

  /// pannel id 
  int pannelName() const { return thePannel; }

  /// plaquetteId (in pannel)
  int plaquetteName() const { return thePlaquette; }

private:
  HalfCylinder thePart;
  int theDisk, theBlade, thePannel, thePlaquette;
};

std::ostream & operator<<( std::ostream& out, const PixelEndcapName::HalfCylinder & t);
#endif
