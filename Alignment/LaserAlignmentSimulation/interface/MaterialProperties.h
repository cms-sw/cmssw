/* 
 * Class to define custom material properties
 */

#ifndef LaserAlignmentSimulation_MaterialProperties_h
#define LaserAlignmentSimulation_MaterialProperties_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4MaterialTable.hh"
#include "G4RegionStore.hh"
#include "G4Region.hh"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4VPhysicalVolume.hh"
#include "G4SDManager.hh"

class MaterialProperties
{
 public:
  MaterialProperties(int DebugLevel, double SiAbsLengthScale);  // constructor
  ~MaterialProperties();    // destructor

 private:
  void setMaterialProperties();

 private:
  const G4MaterialTable * theMaterialTable;

 private:
  int theMPDebugLevel;
  double theSiAbsLengthScalingFactor;
  G4MaterialPropertiesTable * theMPT;
  G4Material * theTECWafer;
  G4Material * theTOBWafer;
  G4Material * theTIBWafer;
};
#endif
