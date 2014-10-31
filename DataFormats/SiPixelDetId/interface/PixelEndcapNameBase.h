#ifndef SiPixelDetId_PixelEndcapNameBase_H
#define SiPixelDetId_PixelEndcapNameBase_H

#include <string>
#include <iostream>

class PixelEndcapNameBase {
public:

  enum HalfCylinder { mO = 1, mI = 2 , pO =3 , pI =4 };

  /// ctor from DetId
//  PixelEndcapNameBase(const DetId &, bool phase=false);
//  PixelEndcapNameBase(const DetId &, const TrackerTopology* tt, bool phase=false);

  PixelEndcapNameBase(HalfCylinder part = mO, int disk =0, int blade =0, int pannel=0, int plaq=0) :
     thePart(part), theDisk(disk), theBlade(blade), thePannel(pannel), thePlaquette(plaq)//, phase1(phase)
  { }

  /// ctor from name string

//  PixelEndcapNameBase(std::string name);

  virtual ~PixelEndcapNameBase() { }

  /// from base class
//  virtual std::string name() const;

  HalfCylinder halfCylinder() const { return thePart; }

  /// disk id
  int diskName() const { return theDisk; }

  /// blade id
  int bladeName() const { return theBlade; }

  /// pannel id
  int pannelName() const { return thePannel; }

  /// plaquetteId (in pannel)
  int plaquetteName() const { return thePlaquette; }

  /// module Type
//   virtual PixelModuleName::ModuleType  moduleType() const = 0;

  /// return DetId
//  virtual PXFDetId getDetId() = 0;

  /// check equality of modules from datamemebers
//  virtual bool operator== (const PixelModuleName &) const = 0;

protected:
  HalfCylinder thePart;
  int theDisk, theBlade, thePannel, thePlaquette;

};

std::ostream & operator<<( std::ostream& out, const PixelEndcapNameBase::HalfCylinder & t);
#endif
