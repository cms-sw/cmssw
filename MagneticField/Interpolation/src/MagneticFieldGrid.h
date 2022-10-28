#ifndef MagneticFieldGrid_H
#define MagneticFieldGrid_H

/** \class MagneticFieldGrid
 *
 * load magnetic field grid from binary file
 * remark: units are either (cm,cm,cm) or (cm,rad,cm)
 *         and Tesla for the magnetic field
 *
 * additional functions either translate indices <-> coordinates,
 * transfer data, or activate the interpolation between grid points
 *
 * \author : <Volker.Drollinger@cern.ch>
 *
 * Modifications:
 *
 */

// interpolation package
#include "FWCore/Utilities/interface/Visibility.h"
#include "VectorFieldInterpolation.h"

// used libs
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>

class dso_internal MagneticFieldGrid {
public:
  // constructor
  MagneticFieldGrid() {
    GridType = 0;
    for (int i = 0; i < 3; ++i) {
      NumberOfPoints[i] = 0;
    };
    for (int i = 0; i < 3; ++i) {
      ReferencePoint[i] = 0.;
    };
    for (int i = 0; i < 3; ++i) {
      BasicDistance0[i] = 0.;
    };
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        BasicDistance1[i][j] = 0.;
      };
    };
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        BasicDistance2[i][j] = 0.;
      };
    };
    for (int i = 0; i < 4; ++i) {
      RParAsFunOfPhi[i] = 0.;
    };
    for (int i = 0; i < 3; ++i) {
      EasyCoordinate[i] = false;
    };
  }
  // destructor
  ~MagneticFieldGrid() {}

private:
  // header classes (5: one for each type)
  class dso_internal HeaderType3 {
  public:
    // constructor
    HeaderType3() {}
    // destructor
    ~HeaderType3() {}

  private:
  public:
    void printInfo();
  };
  // b-field container
  class dso_internal BVector {
  public:
    // constructor
    BVector() {}
    // destructor
    ~BVector() {}

  private:
    // three component vector in float precision
    float B3[3];

  public:
    // Accessors
    void putB3(float Bx, float By, float Bz);
    float bx();
    float by();
    float bz();
  };

  // DEFINITION OF GRID
  // type
  int GridType;
  // header
  int NumberOfPoints[3];
  double ReferencePoint[3];
  double BasicDistance0[3];     // constant step
  double BasicDistance1[3][3];  // linear step
  double BasicDistance2[3][3];  // linear offset
  double RParAsFunOfPhi[4];     // R = f(phi) or const. (0,2: const. par. ; 1,3: const./sin(phi))
  bool EasyCoordinate[3];
  // field (Bx,By,Bz) container
  std::vector<BVector> FieldValues;

public:
  /// load grid binary file
  void load(const std::string &name);
  /// returns value of GridType (and eventually prints the type + short description)
  int gridType();

  /// interpolates the magnetic field at input coordinate point and returns field values
  void interpolateAtPoint(double X1, double X2, double X3, float &Bx, float &By, float &Bz);

  // calculates indices from coordinates
  void putCoordGetInd(double X1, double X2, double X3, int &Index1, int &Index2, int &Index3);
  // takes indices and returns magnetic field values
  void putIndicesGetB(int Index1, int Index2, int Index3, float &Bx, float &By, float &Bz);
  // takes indices, calculates coordinates, and returns coordinates
  void putIndGetCoord(int Index1, int Index2, int Index3, double &X1, double &X2, double &X3);
  // converts three indices into one number (for the vector FieldValues)
  int lineNumber(int Index1, int Index2, int Index3);
};

#endif
