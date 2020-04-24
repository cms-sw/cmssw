#ifndef MFGrid_h
#define MFGrid_h

/** \class MFGrid
 *
 *  Virtual interface for a field provider that is based on interpolation
 *  on a regular grid.
 *
 *  \author T. Todorov
 */

#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include "MagneticField/Interpolation/interface/MagProviderInterpol.h"

struct Dimensions 
{
  int w;
  int h;
  int d;
};

struct Indexes
{
  int i;
  int j;
  int k;
};

class MFGrid : public MagProviderInterpol {
public:

  typedef GloballyPositioned<float>::GlobalPoint     GlobalPoint;
  typedef GloballyPositioned<float>::GlobalVector    GlobalVector;
  typedef GloballyPositioned<float>::LocalPoint      LocalPoint;
  typedef GloballyPositioned<float>::LocalVector     LocalVector;

  explicit MFGrid( const GloballyPositioned<float>& vol) : frame_(vol) {}

  ~MFGrid() override {}

  /// Interpolated field value at given point.
  LocalVector valueInTesla( const LocalPoint& p) const override = 0;

  virtual void dump() const {}

  /// find grid coordinates for point. For debugging and validation only.
  virtual void toGridFrame( const LocalPoint& p, double& a, double& b, double& c) const = 0;

  /// find grid coordinates for point. For debugging and validation only.
  virtual LocalPoint fromGridFrame( double a, double b, double c) const = 0;

  virtual Dimensions dimensions() const = 0;

  /// Position of node in local frame
  virtual LocalPoint  nodePosition( int i, int j, int k) const = 0;

  /// Field value at node
  virtual LocalVector nodeValue( int i, int j, int k) const = 0;

  virtual Indexes index( const LocalPoint& p) const {return Indexes();}

  /// Local reference frame
  const GloballyPositioned<float>& frame() const { return frame_;}

private:

  GloballyPositioned<float> frame_;

};

#endif
