#include "MagneticField/Interpolation/interface/MFGridFactory.h"
#include "MagneticField/Interpolation/interface/binary_ifstream.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"

#include "MagneticField/Interpolation/interface/RectangularCartesianMFGrid.h"
#include "MagneticField/Interpolation/interface/RectangularCylindricalMFGrid.h"
#include "MagneticField/Interpolation/interface/TrapezoidalCartesianMFGrid.h"
#include "MagneticField/Interpolation/interface/TrapezoidalCylindricalMFGrid.h"
#include "MagneticField/Interpolation/interface/SpecialCylindricalMFGrid.h"
#include "MagneticField/Interpolation/interface/CylinderFromSectorMFGrid.h"

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
//    inFile.close();
//    result = new GlobalGridWrapper(vol, name);
    result = new TrapezoidalCylindricalMFGrid(inFile, vol);
    break;
  case 5:
    result = new SpecialCylindricalMFGrid(inFile, vol);
    //    inFile.close();
    //    result = new GlobalGridWrapper(vol, name);
    break;
  default:
    cout << "Grid type unknown" << endl;
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
