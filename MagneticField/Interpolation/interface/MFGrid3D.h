#ifndef MFGrid3D_H
#define MFGrid3D_H

#include "MagneticField/Interpolation/interface/MFGrid.h"
#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include "MagneticField/Interpolation/interface/Grid1D.h"
#include "MagneticField/Interpolation/interface/Grid3D.h"

class MFGrid3D : public MFGrid {
public:

  explicit MFGrid3D( const GloballyPositioned<float>& vol) : MFGrid(vol) {}


  virtual std::vector<int> dimensions() const {
    std::vector<int> result(3);
    result[0] = grid_.grida().nodes();
    result[1] = grid_.gridb().nodes();
    result[2] = grid_.gridc().nodes();
    return result;
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

  virtual std::vector<int> index( const LocalPoint& p) const {
    std::vector<int> result(3);
    double a, b, c;
    toGridFrame( p, a, b, c);
    result[0] = grid_.grida().index(a);
    result[1] = grid_.gridb().index(b);
    result[2] = grid_.gridc().index(c);
    return result;
  }

  virtual LocalVector valueInTesla( const LocalPoint& p) const;
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
