#include "MagneticField/Interpolation/interface/MFGridFactory.h"
#include "MagneticField/Interpolation/interface/binary_ifstream.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"

#include "RectangularCartesianMFGrid.h"
#include "RectangularCylindricalMFGrid.h"
#include "TrapezoidalCartesianMFGrid.h"
#include "TrapezoidalCylindricalMFGrid.h"
#include "SpecialCylindricalMFGrid.h"
#include "CylinderFromSectorMFGrid.h"

#include <iostream>

using namespace std;

MFGrid* MFGridFactory::build(const string& name, const GloballyPositioned<float>& vol) {
  magneticfield::interpolation::binary_ifstream inFile(name);
  return build(inFile, vol);
}

MFGrid* MFGridFactory::build(binary_ifstream& inFile, const GloballyPositioned<float>& vol) {
  int gridType;
  inFile >> gridType;

  MFGrid* result;
  switch (gridType) {
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
      break;
    case 6:
      result = new SpecialCylindricalMFGrid(inFile, vol, gridType);
      break;
    default:
      cout << "ERROR Grid type unknown: " << gridType << endl;
      //    result = new GlobalGridWrapper(vol, name);
      result = nullptr;
      break;
  }
  return result;
}

MFGrid* MFGridFactory::build(const string& name, const GloballyPositioned<float>& vol, double phiMin, double phiMax) {
  MFGrid* sectorGrid = build(name, vol);
  return new CylinderFromSectorMFGrid(vol, phiMin, phiMax, sectorGrid);
}
