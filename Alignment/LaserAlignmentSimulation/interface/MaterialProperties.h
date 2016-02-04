#ifndef LaserAlignmentSimulation_MaterialProperties_h
#define LaserAlignmentSimulation_MaterialProperties_h

/** \class MaterialProperties
 *  Class to define custom material properties
 *
 *  $Date: 2007/06/11 14:44:28 $
 *  $Revision: 1.3 $
 *  \author Maarten Thomas
 */

#include "G4MaterialPropertiesTable.hh"
#include "G4SDManager.hh"

class MaterialProperties
{
 public:
	/// constructor
  MaterialProperties(int DebugLevel, double SiAbsLengthScale);
	/// destructor
  ~MaterialProperties();

 private:
	/// define optical properties of materials in the detector
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
