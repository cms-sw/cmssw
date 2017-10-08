#ifndef MFGrid3D_h
#define MFGrid3D_h

/** \class MFGrid3D
 *
 *  Generic virtual implementation of a MFGrid for a 3D underlying regular grid.
 *
 *  \author T. Todorov
 */

#include "MagneticField/Interpolation/interface/MFGrid.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "Grid1D.h"
#include "Grid3D.h"
#include "FWCore/Utilities/interface/Visibility.h"

class dso_internal MFGrid3D : public MFGrid {
public:

  explicit MFGrid3D( const GloballyPositioned<float>& vol) : MFGrid(vol) {}


  Dimensions dimensions(void) const override {
    Dimensions tmp;
    tmp.w = grid_.grida().nodes();
    tmp.h = grid_.gridb().nodes();
    tmp.d = grid_.gridc().nodes();
    return tmp;
  }
    
  /// Position of node in local frame
  LocalPoint  nodePosition( int i, int j, int k) const override {
    return fromGridFrame( grid_.grida().node(i), grid_.gridb().node(j), grid_.gridc().node(k));
  }

  /// Field value at node
  LocalVector nodeValue( int i, int j, int k) const override {
    /// must check range here: FIX ME !!!!
    return MFGrid::LocalVector(grid_( i, j, k));
  }

  Indexes index( const LocalPoint& p) const override {
    Indexes result;
    double a, b, c;
    toGridFrame( p, a, b, c);
    result.i = grid_.grida().index(a);
    result.j = grid_.gridb().index(b);
    result.k = grid_.gridc().index(c);
    return result;
  }

  LocalVector valueInTesla( const LocalPoint& p) const override;

  /// Interpolated field value at given point; does not check for exceptions
  virtual LocalVector uncheckedValueInTesla( const LocalPoint& p) const = 0;

protected:

  using GridType =  Grid3D;
  using BVector = Grid3D::BVector;

  GridType       grid_; // should become private...

  void setGrid( const GridType& grid) {
    grid_ = grid;
  }

};

#endif
