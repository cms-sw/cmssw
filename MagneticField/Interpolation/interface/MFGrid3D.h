#ifndef MFGrid3D_h
#define MFGrid3D_h

/** \class MFGrid3D
 *
 *  Generic virtual implementation of a MFGrid for a 3D underlying regular grid.
 *
 *  $Date: 2008/04/09 19:37:23 $
 *  $Revision: 1.7 $
 *  \author T. Todorov
 */

#include "MagneticField/Interpolation/interface/MFGrid.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "MagneticField/Interpolation/src/Grid1D.h"
#include "MagneticField/Interpolation/src/Grid3D.h"

class MFGrid3D : public MFGrid {
public:

  explicit MFGrid3D( const GloballyPositioned<float>& vol) : MFGrid(vol) {}


  virtual Dimensions dimensions(void) const {
    Dimensions tmp;
    tmp.w = grid_.grida().nodes();
    tmp.h = grid_.gridb().nodes();
    tmp.d = grid_.gridc().nodes();
    return tmp;
  }
    
  /// Position of node in local frame
  virtual LocalPoint  nodePosition( int i, int j, int k) const {
    return fromGridFrame( grid_.grida().node(i), grid_.gridb().node(j), grid_.gridc().node(k));
  }

  /// Field value at node
  virtual LocalVector nodeValue( int i, int j, int k) const {
    /// must check range here: FIX ME !!!!
    return MFGrid::LocalVector(grid_( i, j, k));
  }

  virtual Indexes index( const LocalPoint& p) const {
    Indexes result;
    double a, b, c;
    toGridFrame( p, a, b, c);
    result.i = grid_.grida().index(a);
    result.j = grid_.gridb().index(b);
    result.k = grid_.gridc().index(c);
    return result;
  }

  virtual LocalVector valueInTesla( const LocalPoint& p) const;

  /// Interpolated field value at given point; does not check for exceptions
  virtual LocalVector uncheckedValueInTesla( const LocalPoint& p) const = 0;

protected:

  typedef Basic3DVector<float>      BVector;
  typedef Grid3D< BVector, double>  GridType;

  GridType       grid_; // should become private...

  void setGrid( const GridType& grid) {
    grid_ = grid;
  }

};

#endif
