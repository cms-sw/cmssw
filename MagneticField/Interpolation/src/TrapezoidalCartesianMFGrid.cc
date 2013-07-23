/** \file
 *
 *  $Date: 2011/04/16 12:47:37 $
 *  $Revision: 1.8 $
 *  \author T. Todorov - updated N. Amapane (2008)
 */

#include "TrapezoidalCartesianMFGrid.h"
#include "binary_ifstream.h"
#include "LinearGridInterpolator3D.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"
#include <iostream>

//#define DEBUG_GRID

using namespace std;

TrapezoidalCartesianMFGrid::TrapezoidalCartesianMFGrid( binary_ifstream& inFile,
							const GloballyPositioned<float>& vol)
  : MFGrid3D(vol), increasingAlongX(false), convertToLocal(true)
{
  
  // The parameters read from the data files are given in global coordinates.
  // In version 85l, local frame has the same orientation of global frame for the reference
  // volume, i.e. the r.f. transformation is only a translation.
  // There is therefore no need to convert the field values to local coordinates.

  // Check orientation of local reference frame: 
  GlobalVector localXDir(frame().toGlobal(LocalVector(1,0,0)));
  GlobalVector localYDir(frame().toGlobal(LocalVector(0,1,0)));

  if (localXDir.dot(GlobalVector(1,0,0)) > 0.999999 &&
      localYDir.dot(GlobalVector(0,1,0)) > 0.999999) {
    // "null" rotation - requires no conversion...
    convertToLocal = false;
  } else if (localXDir.dot(GlobalVector(0,1,0)) > 0.999999 &&
	     localYDir.dot(GlobalVector(1,0,0)) > 0.999999) {
    // Typical orientation if master volume is in sector 1 
    convertToLocal = true;    
  } else {
    convertToLocal = true;    
    // Nothing wrong in principle, but this is not expected
    cout << "WARNING: TrapezoidalCartesianMFGrid: unexpected orientation: x: " 
	 << localXDir << " y: " << localYDir << endl;
  }

  int n1, n2, n3;
  inFile >> n1 >> n2 >> n3;
  double xref, yref, zref;
  inFile >> xref >> yref >> zref;
  double step1, step2, step3;
  inFile >> step1    >> step2    >> step3;

  double BasicDistance1[3][3];  // linear step
  double BasicDistance2[3][3];  // linear offset
  bool   easya, easyb, easyc;

  inFile >> BasicDistance1[0][0] >> BasicDistance1[1][0] >> BasicDistance1[2][0];
  inFile >> BasicDistance1[0][1] >> BasicDistance1[1][1] >> BasicDistance1[2][1];
  inFile >> BasicDistance1[0][2] >> BasicDistance1[1][2] >> BasicDistance1[2][2];
  inFile >> BasicDistance2[0][0] >> BasicDistance2[1][0] >> BasicDistance2[2][0];
  inFile >> BasicDistance2[0][1] >> BasicDistance2[1][1] >> BasicDistance2[2][1];
  inFile >> BasicDistance2[0][2] >> BasicDistance2[1][2] >> BasicDistance2[2][2];
  inFile >> easya >> easyb >> easyc;

  vector<BVector> fieldValues;
  float Bx, By, Bz;
  int nLines = n1*n2*n3;
  fieldValues.reserve(nLines);
  for (int iLine=0; iLine<nLines; ++iLine){
    inFile >> Bx >> By >> Bz;
    if (convertToLocal) {
      // Preserve double precision!
      Vector3DBase<double, LocalTag>  lB = frame().toLocal(Vector3DBase<double, GlobalTag>(Bx,By,Bz));
      fieldValues.push_back(BVector(lB.x(), lB.y(), lB.z()));
      
    } else {
      fieldValues.push_back(BVector(Bx,By,Bz));
    }
  }
  // check completeness
  string lastEntry;
  inFile >> lastEntry;
  if (lastEntry != "complete") {
    cout << "ERROR during file reading: file is not complete" << endl;
  }

  // In version 1103l the reference sector is at phi=0 so that local y is along global X.
  // The increasing grid steps for the same volume are then along Y instead than along X.
  // To use Trapezoid2RectangleMappingX to map the trapezoidal geometry to a rectangular
  // cartesian geometry, we have to exchange (global) X and Y appropriately when constructing
  // it.
  int nx, ny;
  double stepx, stepy, stepz;
  double dstep, offset;
  if (!easya && easyb && easyc) {
    // Increasing grid spacing is on x
    increasingAlongX = true;
    nx = n1;
    ny = n2;
    stepx = step1; 
    stepy = step2;
    stepz = step3;
    dstep = BasicDistance1[0][1];
    offset = BasicDistance2[0][1];
  } else if (easya && !easyb && easyc) {
    // Increasing grid spacing is on y
    increasingAlongX = false;
    nx = n2;
    ny = n1;
    stepx = step2; 
    stepy = step1;
    stepz = -step3;
    dstep = BasicDistance1[1][0];
    offset = BasicDistance2[1][0];

   } else {
    // Increasing spacing on z or on > 1 coordinate not supported
    throw MagGeometryError("TrapezoidalCartesianMFGrid only implemented for first or second coordinate");
  }

  double a = stepx * (nx -1);             // first base
  double b = a + dstep * (ny-1) * (nx-1); // second base
  double h = stepy * (ny-1);              // height
  double delta = -offset * (ny-1);        // offset between two bases
  double baMinus1 = dstep*(ny-1) / stepx; // (b*a) - 1

  GlobalPoint grefp( xref, yref, zref);
  LocalPoint lrefp = frame().toLocal( grefp);

  if (fabs(baMinus1) > 0.000001) {
    double b_over_a = 1 + baMinus1;
    double a1 = delta/baMinus1;

#ifdef DEBUG_GRID
    cout << "Trapeze size (a,b,h) = " << a << "," << b << "," << h << endl;
    cout << "Global origin " << grefp << endl;
    cout << "Local origin  " << lrefp << endl;
    cout << "a1 = " << a1 << endl;
#endif

    // FIXME ASSUMPTION: here we assume that the local reference frame is oriented with X along 
    // the direction of where the grid is not uniform. This is the case for all current geometries
    double x0 = lrefp.x() + a1;
    double y0 = lrefp.y() + h/2.;
    mapping_ = Trapezoid2RectangleMappingX( x0, y0, b_over_a, h);
  }
  else { // parallelogram
    mapping_ = Trapezoid2RectangleMappingX( 0, 0, delta/h);
  }

  // transform reference point to grid frame
  double xrec, yrec;
  mapping_.rectangle( lrefp.x(), lrefp.y(), xrec, yrec);

  Grid1D gridX( xrec, xrec + (a+b)/2., nx);
  Grid1D gridY( yrec, yrec + h, ny);
  Grid1D gridZ( lrefp.z(), lrefp.z() + stepz*(n3-1), n3);  

#ifdef DEBUG_GRID
  cout << " GRID X range: local " << gridX.lower() <<  " - " << gridX.upper()
       <<" global: " << (frame().toGlobal(LocalPoint(gridX.lower(),0,0))).y() << " - " 
       << (frame().toGlobal(LocalPoint(gridX.upper(),0,0))).y() << endl;

  cout << " GRID Y range: local " << gridY.lower() <<  " - " << gridY.upper()
       << " global: " << (frame().toGlobal(LocalPoint(0,gridY.lower(),0))).x() << " - " 
       << (frame().toGlobal(LocalPoint(0,gridY.upper(),0))).x() << endl;

  cout << " GRID Z range: local " << gridZ.lower() <<  " - " << gridZ.upper()
       << " global: " << (frame().toGlobal(LocalPoint(0,0,gridZ.lower()))).z() << " " 
       << (frame().toGlobal(LocalPoint(0,0,gridZ.upper()))).z() << endl;
#endif

  if (increasingAlongX) {
    grid_ = GridType( gridX, gridY, gridZ, fieldValues);
  } else {
    // The reason why gridY and gridX have to be exchanged is because Grid3D::index(i,j,k)
    // assumes a specific order for the fieldValues, and we cannot rearrange this vector.
    // Given that we exchange grids, we will have to exchange the outpouts of mapping_rectangle()
    // and the inputs of mapping_.trapezoid() in the following...
    grid_ = GridType( gridY, gridX, gridZ, fieldValues);
  }
    
  
#ifdef DEBUG_GRID
  dump();
#endif
}

void TrapezoidalCartesianMFGrid::dump() const
{
  
  cout << endl << "Dump of TrapezoidalCartesianMFGrid" << endl;
//   cout << "Number of points from file   " 
//        << n1 << " " << n2 << " " << n3 << endl;
  cout << "Number of points from Grid1D " 
       << grid_.grida().nodes() << " " << grid_.gridb().nodes() << " " << grid_.gridc().nodes() << endl;

//   cout << "Reference Point from file   " 
//        << xref << " " << yref << " " << zref << endl;
  cout << "Reference Point from Grid1D " 
       << grid_.grida().lower() << " " << grid_.gridb().lower() << " " << grid_.gridc().lower() << endl;

//   cout << "Basic Distance from file   " 
//        <<  stepx << " " << stepy << " " << stepz << endl;
  cout << "Basic Distance from Grid1D "
       << grid_.grida().step() << " " << grid_.gridb().step() << " " << grid_.gridc().step() << endl;


  cout << "Dumping " << grid_.data().size() << " field values " << endl;
  // grid_.dump();
  

  // Dump ALL grid points and values
  // CAVEAT: if convertToLocal = true in the ctor, points have been converted to LOCAL
  // coordinates. To match those from .table files they have to be converted back to global
//   for (int j=0; j < grid_.gridb().nodes(); j++) {
//     for (int i=0; i < grid_.grida().nodes(); i++) {
//       for (int k=0; k < grid_.gridc().nodes(); k++) {
// 	cout << i << " " << j << " " << k << " "  
// 	     << frame().toGlobal(LocalPoint(nodePosition(i,j,k))) << " "
// 	     << nodeValue(i,j,k) << endl;
//       }
//     }
//   }
  

}

MFGrid::LocalVector TrapezoidalCartesianMFGrid::uncheckedValueInTesla( const LocalPoint& p) const
{

  double xrec, yrec;
  mapping_.rectangle( p.x(), p.y(), xrec, yrec);
  

// cout << "TrapezoidalCartesianMFGrid::valueInTesla at local point " << p << endl;
//   cout << p.x() << " " << p.y()  
//        << " transformed to grid frame: " << xrec << " " << yrec << endl;

  LinearGridInterpolator3D interpol( grid_);

  if (!increasingAlongX) {
    std::swap(xrec,yrec);
    // B values are already converted to local coord!!! otherwise we should:
    // GridType::ValueType value = interpol.interpolate( xrec, yrec, p.z());
    // return LocalVector(value.y(),value.x(),-value.z());
  }

  return LocalVector(interpol.interpolate( xrec, yrec, p.z()));
}

void TrapezoidalCartesianMFGrid::toGridFrame( const LocalPoint& p, 
					      double& a, double& b, double& c) const
{
  if (increasingAlongX) {
    mapping_.rectangle( p.x(), p.y(), a, b);
  } else {
    mapping_.rectangle( p.x(), p.y(), b, a);
  }  
  
  c = p.z();
}
 
MFGrid::LocalPoint TrapezoidalCartesianMFGrid::fromGridFrame( double a, double b, double c) const
{
  double xtrap, ytrap;
  if (increasingAlongX) {
    mapping_.trapezoid( a, b, xtrap, ytrap);
  } else {    
    mapping_.trapezoid( b, a, xtrap, ytrap);
  }
  return LocalPoint( xtrap, ytrap, c);
}
