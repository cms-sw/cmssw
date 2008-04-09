#include "MagneticField/Interpolation/interface/MFGridFactory.h"
#include "MagneticField/Interpolation/src/binary_ifstream.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"

#include "MagneticField/Interpolation/src/RectangularCartesianMFGrid.h"
#include "MagneticField/Interpolation/src/RectangularCylindricalMFGrid.h"
#include "MagneticField/Interpolation/src/TrapezoidalCartesianMFGrid.h"
#include "MagneticField/Interpolation/src/TrapezoidalCylindricalMFGrid.h"
#include "MagneticField/Interpolation/src/SpecialCylindricalMFGrid.h"
#include "MagneticField/Interpolation/src/CylinderFromSectorMFGrid.h"

#include <iostream>

using namespace std;

MFGrid* MFGridFactory::build(const string& name, const GloballyPositioned<float>& vol) {
  binary_ifstream inFile(name);
  int gridType;
  inFile >> gridType;

  MFGrid* result;
  switch (gridType){
  case 1:
    result = new RectangularCartesianMFGrid(inFile, vol);
    break;
  case 2:
    result = new TrapezoidalCartesianMFGrid(inFile, vol);
    break;
  case 3:
    result = new RectangularCylindricalMFGrid(inFile, vol);
    break;
  case 4:
    result = new TrapezoidalCylindricalMFGrid(inFile, vol);
    break;
  case 5:
    result = new SpecialCylindricalMFGrid(inFile, vol, gridType);
  case 6:
    result = new SpecialCylindricalMFGrid(inFile, vol, gridType);
    break;
  default:
    cout << "ERROR Grid type unknown: " << gridType << endl;
    //    result = new GlobalGridWrapper(vol, name);
    result = 0;
    break;
  }
  inFile.close();
  return result;
}

MFGrid* MFGridFactory::build(const string& name, 
			     const GloballyPositioned<float>& vol,
			     double phiMin, double phiMax)
{
  MFGrid* sectorGrid = build(name,vol);
  return new CylinderFromSectorMFGrid( vol, phiMin, phiMax, sectorGrid);
}
