// H4Geom.h
//
// Class which manages geometry information of the super-module
//
// last change : $Date: 2004/10/01 16:38:10 $
// by          : $Author: brunelie $
//

#ifndef H4Geom_H
#define H4Geom_H


#include "TObject.h"
#include <string>

class H4Geom: public TObject
{
 public:
  // Geometry is period dependent. The different geometries are listed below
  enum GeomPeriod_t {
    Undef,      // Cause the program to crash
    Year2003,   // Test beam during 2003: SM0 and SM1
    Spring2004, // June-July 2004: E0' tests
    Automn2004  // Tests of one full supermodule
  };

  // Constants defining supermodule geometry
  enum SMGeom_t {
    kModules = 4,           // Number of modules per supermodule
    kTriggerTowers = 68,    // Number of trigger towers per supermodule
    kTowersInPhi = 4,       // Number of trigger towers in phi
    kTowersInEta = 17,      // Number of trigger towers in eta
    kCrystals = 1700,       // Number of crystals per supermodule
    kCrystalsInPhi = 20,    // Number of crystals in phi
    kCrystalsInEta = 85,    // Number of crystals in eta
    kCrystalsPerTower = 25, // Number of crystals per trigger tower
    kCardsPerTower = 5,     // Number of VFE cards per trigger tower
    kChannelsPerCard = 5    // Number of channels per VFE card
  };

  // Default Constructor, mainly for Root
  H4Geom() ;

  // Destructor: Does nothing?
  virtual ~H4Geom();

  // Initialize geometry with config file
  bool init();

  // Retuns the crystal number in the super module for a given
  // tower number in the super module and crystal number in the tower
  int getSMCrystalNumber(int tower, int crystal) const ;

  // Retuns the crystal number in a tower for a given
  // crystal number in the super module
  void getTowerCrystalNumber(int &tower, int &crystal, int smCrystal) const ;

  // Returns the crystal number (readout order) in a tower 
  // for a given position in the tower (crystalNbGeom=0 is the 
  // lower-right corner and crystalNbGeom=24 is the upper-left corner)
  int getTowerCrystalNumber(int smTowerNb, int crystalNbGeom) const ;

  // Returns the crystal coordinates (eta, phi index) for a given
  // crystal number in the super module
  void getCrystalCoord(int &eta, int &phi, int smCrystal) const ;

  // Retuns the crystal number in the super module for given coordinates
  int getSMCrystalFromCoord(int eta, int phi) const ;

  // Returns left neighbour of a sm crystal.
  // Input and output are crystal numbers in the super module.
  // A negative output means outside of the supermodule.
  int getLeft(int smCrystal) const ;

  // Returns right neighbour of a sm crystal.
  // Input and output are crystal numbers in the super module.
  // A negative output means outside of the supermodule.
  int getRight(int smCrystal) const ;

  // Returns upper neighbour of a sm crystal.
  // Input and output are crystal numbers in the super module.
  // A negative output means outside of the supermodule.
  int getUpper(int smCrystal) const ;

  // Returns lower neighbour of a sm crystal.
  // Input and output are crystal numbers in the super module.
  // A negative output means outside of the supermodule.
  int getLower(int smCrystal) const ;

  // Returns left neighbour of a crystal referenced by its coordinates.
  // New coordonates overwrite the old ones. No check is done to see
  // if it corresponds to a real crystal. To be used with caution. 
  void mvLeft(int &eta, int &phi) const ;

  // Returns right neighbour of a crystal referenced by its coordinates.
  // New coordonates overwrite the old ones. No check is done to see
  // if it corresponds to a real crystal. To be used with caution. 
  void mvRight(int &eta, int &phi) const ;

  // Returns upper neighbour of a crystal referenced by its coordinates.
  // New coordonates overwrite the old ones. No check is done to see
  // if it corresponds to a real crystal. To be used with caution. 
  void mvUp(int &eta, int &phi) const ;


  // Returns lower neighbour of a crystal referenced by its coordinates.
  // New coordonates overwrite the old ones. No check is done to see
  // if it corresponds to a real crystal. To be used with caution. 
  void mvDown(int &eta, int &phi) const ;

  // Returns the 25 crystals of tower towerNb in the super module.
  // Output are crystal numbers in the super module.
  // By default, the order in the output array (tower) corresponds to
  // geometric order (index 0 is lower-right corner).
  // if order=readout, the order in the output array (tower) 
  // corresponds to the readout scheme (depends on the kind of tower) 
  void getTower(int * tower, int towerNb, std::string order = "geom") const ;

  // Returns the 5 crystals belonging to the same VFE board as smCrystal.
  // Input and output are crystal numbers in the super module.
  // By default, the order in the output array (VFE) corresponds to
  // The geometric order (index 0 is lower-right corner).
  // if order=readout, the order in the output array (VFE) 
  // corresponds to the readout scheme (depends on the kind of tower)
  void getVFE(int * VFE, int smCrystal, std::string order = "geom") const ;

  // Returns sm crystal numbers for crystals in a window of
  // size widthxheight centered around a given smCrystal.
  // width and height must be odd.
  // The order in the output array (window) is defined 
  // by the geometric order (index 0 is lower-right corner).
  void getWindow(int * window, int smCrystal, int width, int height) const ;

  // Tests if low voltage board is on the right size of the tower. 
  // Readout scheme depends on that.
  bool rightTower(int tower) const ;

  // Tests if low voltage board is on the left size of the tower
  // Readout scheme depends on that.
  bool leftTower(int tower) const ;
 
  GeomPeriod_t GetGeomPeriod() const {return geometry_;}

  static void SetGeomPeriod(GeomPeriod_t geometry);

 private:
  bool IsGeomPeriodDefined() const;

  const static int crystalChannelMap[5][5];
  const static int crystalMap[25];
  static GeomPeriod_t geometry_;

  ClassDef(H4Geom,1) // Class to navigate in a SM
};

#endif
