#ifndef Geometry_CommonTopologies_PixelTopology_H
#define Geometry_CommonTopologies_PixelTopology_H

#include "Geometry/CommonTopologies/interface/Topology.h"

/**   
 * Interface for all pixel topologies  
 */

class PixelTopology : public Topology {
public:
  ////explicit PixelTopology( DetType* d) : Topology(d) {}
  ~PixelTopology() override {}

  //  The following methods are moved to the base class (Topology)

  //  virtual LocalPoint local_position( float channel) const = 0;
  //  virtual LocalError local_error( float err) const = 0;
  //  virtual int channel( const LocalPoint& p) const = 0;

  virtual std::pair<float, float> pixel(const LocalPoint &p) const = 0;

  /// conversion taking also the angle from the track state
  virtual std::pair<float, float> pixel(const LocalPoint &p, const Topology::LocalTrackAngles & /*ltp*/) const {
    return pixel(p);
  }

  virtual std::pair<float, float> pitch() const = 0;
  virtual int nrows() const = 0;
  virtual int ncolumns() const = 0;

  virtual int rocsY() const = 0;
  virtual int rocsX() const = 0;
  virtual int rowsperroc() const = 0;
  virtual int colsperroc() const = 0;

  virtual float localX(float mpX) const = 0;
  virtual float localY(float mpY) const = 0;
  virtual float localX(const float mpX, const Topology::LocalTrackPred & /*trk*/) const { return localX(mpX); }
  virtual float localY(const float mpY, const Topology::LocalTrackPred & /*trk*/) const { return localY(mpY); }

  virtual bool isItBigPixelInX(int ixbin) const = 0;
  virtual bool isItBigPixelInY(int iybin) const = 0;
  virtual bool containsBigPixelInX(int ixmin, int ixmax) const = 0;
  virtual bool containsBigPixelInY(int iymin, int iymax) const = 0;

  virtual bool isItEdgePixelInX(int ixbin) const = 0;
  virtual bool isItEdgePixelInY(int iybin) const = 0;
  virtual bool isItEdgePixel(int ixbin, int iybin) const = 0;
};

#endif
