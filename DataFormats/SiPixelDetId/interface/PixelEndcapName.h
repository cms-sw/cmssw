#ifndef PixelEndcapName_H
#define PixelEndcapName_H

/** \class PixelEndcapName
 * Endcap Module name (as in PixelDatabase) for endcaps
 */

#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"

#include <string>

class DetId;

class PixelEndcapName : public PixelModuleName {
public:

  /// ctor from DetId
  PixelEndcapName(const DetId &);
  
  /// ctor for defined name
  PixelEndcapName(int endcap, int disk, int blade, int pannel, int plaquette) 
    : PixelModuleName(false), theEndCap(endcap), theDisk(disk), 
      theBlade(blade), thePannel(pannel), thePlaquette(plaquette)
  { }

  virtual ~PixelEndcapName() { }

  /// from base class
  virtual std::string name() const;


  /// endcap id
  int endcapName() const { return theEndCap; }

  /// disk id
  int diskName() const { return theDisk; }

  /// blade id
  int bladeName() const { return theBlade; }

  /// pannel id 
  int pannelName() const { return thePannel; }

  /// plaquetteId (in pannel)
  int plaquetteName() const { return thePlaquette; }

private:
  int theEndCap, theDisk, theBlade, thePannel, thePlaquette;
};
#endif
