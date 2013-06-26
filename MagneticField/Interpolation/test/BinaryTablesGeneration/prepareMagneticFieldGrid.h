#ifndef prepareMagneticFieldGrid_H
#define prepareMagneticFieldGrid_H

/** \class prepareMagneticFieldGrid
 *
 * read coordinates and magnetic field values from an ASCII file
 *
 * The file given as inmust be in the form:
 *   *-xyz-* (Cartesian) 
 *   *-rpz-* (Cylindrical)
 * -remark: the units of the ASCII file are unknown to this class
 *
 * Inputs are assumed to be in local coordinates if rotateFromSector=0.
 * Otherwise, inputs are assumed to be in global coordinates, and are converted 
 * to the local reference frame of the sector specified in rotateFromSector.
 *
 * determine the structure of the coordinate points (e.g. trapezoid)
 * and store the information on a grid like structure for fast access
 * -remark: one variable may depend linearly on the others 
 * -remark: points do not need to be ordered in ASCII file 
 *
 * additional functions either translate indices <-> coordinates,
 * transfer data, or activate the interpolation between grid points
 *
 * \author : <Volker.Drollinger@cern.ch>, updated N. Amapane 04/2008, 2102, 2013
 * $date   : 09/09/2003 11:49:38 CET $
 * 
 *
 */

// used libs
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>

#define PRINT true
#define EPSILON 1e-4

class prepareMagneticFieldGrid{
public:
  // constructor
  prepareMagneticFieldGrid(int rotateFromSector=0) :
    rotateSector(rotateFromSector)
{
    GridType = 0;
    for (int i=0;i<3; ++i) {NumberOfPoints[i] = 0;};
    for (int i=0;i<3; ++i) {ReferencePoint[i] = 0.;};
    for (int i=0;i<3; ++i) {BasicDistance0[i] = 0.;};
    for (int i=0;i<3; ++i) {for (int j=0;j<3; ++j) {BasicDistance1[i][j] = 0.;};};
    for (int i=0;i<3; ++i) {for (int j=0;j<3; ++j) {BasicDistance2[i][j] = 0.;};};
    for (int i=0;i<4; ++i) {RParAsFunOfPhi[i] = 0.;};
    for (int i=0;i<3; ++i) {EasyCoordinate[i] = false;};
    KnownStructure = false;
    XyzCoordinates = false;
    RpzCoordinates = false;
    sector = unknown;
}
  // destructor
  ~prepareMagneticFieldGrid(){}

private:

  enum masterSector {unknown=0, one=1, four=4};

  class IndexedDoubleVector{
  public:
    // constructor
    IndexedDoubleVector(){}
    // destructor
    ~IndexedDoubleVector(){}
    // less operator (defined for indices: 1st index = highest priority, 2nd index = 2nd highest priority, ...)
    bool operator<(const IndexedDoubleVector&) const;
  private:
    // six component vector in double precision
    int    I3[3];
    double V6[6];
  public:
    // Accessors
    void putI3(int  index1, int  index2, int  index3);
    void getI3(int &index1, int &index2, int &index3);
    void putV6(double  X1, double  X2, double  X3, double  Bx, double  By, double  Bz);
    void getV6(double &X1, double &X2, double &X3, double &Bx, double &By, double &Bz);
  };
  class SixDPoint{
  public:
    // constructor
    SixDPoint(){}
    // destructor
    ~SixDPoint(){}
  private:
    // six component vector in double precision
    double P6[6];
  public:
    // Accessors
    void putP6(double  X1, double  X2, double  X3, double  Bx, double  By, double  Bz);
    double x1();
    double x2();
    double x3();
    double bx();
    double by();
    double bz();
  };
  // definition of grid
  // structure
  int    GridType;
  int    NumberOfPoints[3];
  double ReferencePoint[3];
  double BasicDistance0[3];     // constant step
  double BasicDistance1[3][3];  // linear step
  double BasicDistance2[3][3];  // linear offset
  double RParAsFunOfPhi[4];     // R = f(phi) or const. (0,2: const. par. ; 1,3: const./sin(phi))
  bool   EasyCoordinate[3];
  bool   KnownStructure;
  bool   XyzCoordinates;
  bool   RpzCoordinates;
  int rotateSector;
  masterSector sector;

  // all points (X1,X2,X3,Bx,By,Bz) of one volume
  std::vector<SixDPoint> GridData;

  // convert original units [m] to new units [cm]
  void convertUnits();

public:
  /// check, if number of lines corresponds to number of points in space (double counting test)
  void countTrueNumberOfPoints(const std::string& name) const;
  /// reads the corresponding ASCII file, detects the logic of the points, and saves them on a grid
  void fillFromFile(const std::string& name);
  /// sames as fillFromFile, but for special cases which are not covered by the standard algorithm
  void fillFromFileSpecial(const std::string& name);
  /// returns value of GridType (and eventually prints the type + short description)
  int gridType();
  /// possibility to check existing MagneticFieldGrid point by point (all points)
  void validateAllPoints();
  /// sames as fillFromFile, but for special cases which are not covered by the standard algorithm
  void saveGridToFile(const std::string& outName);

  /// indicates, that MagneticFieldGrid is fully operational (for interpolation)
  bool isReady();

  /// interpolates the magnetic field at input coordinate point and returns field values
  void interpolateAtPoint(double X1, double X2, double X3, double &Bx, double &By, double &Bz);

  // calculates indices from coordinates
  void putCoordGetIndices(double X1, double X2, double X3, int &Index1, int &Index2, int &Index3);
  // takes indices and returns coordinates and magnetic field values 
  void putIndicesGetXAndB(int Index1, int Index2, int Index3, double &X1, double &X2, double &X3, double &Bx, double &By, double &Bz);
  // takes indices, calculates coordinates, and returns coordinates + magnetic fiels values (for testing purposes only)
  void putIndCalcXReturnB(int Index1, int Index2, int Index3, double &X1, double &X2, double &X3, double &Bx, double &By, double &Bz);
  // converts three indices into one number (for the vector GridData)
  int lineNumber(int Index1, int Index2, int Index3);

};


#endif
