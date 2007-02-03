#ifndef MFGrid_H
#define MFGrid_H

#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include "MagneticField/Interpolation/interface/MagProviderInterpol.h"

#include <vector>

class MFGrid : public MagProviderInterpol {
public:

  typedef GloballyPositioned<float>::GlobalPoint     GlobalPoint;
  typedef GloballyPositioned<float>::GlobalVector    GlobalVector;
  typedef GloballyPositioned<float>::LocalPoint      LocalPoint;
  typedef GloballyPositioned<float>::LocalVector     LocalVector;

  explicit MFGrid( const GloballyPositioned<float>& vol) : frame_(vol) {}

  virtual ~MFGrid() {}

  virtual LocalVector valueInTesla( const LocalPoint& p) const = 0;

  virtual void dump() const {}

  /** find grid coordinates for point. For debugging and validation only.
   */
  virtual void toGridFrame( const LocalPoint& p, double& a, double& b, double& c) const = 0;

  /** find grid coordinates for point. For debugging and validation only.
   */
  virtual LocalPoint fromGridFrame( double a, double b, double c) const = 0;

  virtual std::vector<int> dimensions() const = 0;

  /// Position of node in local frame
  virtual LocalPoint  nodePosition( int i, int j, int k) const = 0;

  /// Field value at node
  virtual LocalVector nodeValue( int i, int j, int k) const = 0;

  virtual std::vector<int> index( const LocalPoint& p) const {return std::vector<int>();}

  const GloballyPositioned<float>& frame() const { return frame_;}

private:

  GloballyPositioned<float> frame_;

};

#endif
