#ifndef Geometry_CommonTopologies_PixelTopology_H
#define Geometry_CommonTopologies_PixelTopology_H

#include "Geometry/CommonTopologies/interface/Topology.h"

/**   
 * Interface for all pixel topologies  
 */  

class PixelTopology : public Topology {
 public:  
  ////explicit PixelTopology( DetType* d) : Topology(d) {}
  virtual ~PixelTopology() {}
  
  //  The following methods are moved to the base class (Topology)
  
  //  virtual LocalPoint local_position( float channel) const = 0; 
  //  virtual LocalError local_error( float err) const = 0;
  //  virtual int channel( const LocalPoint& p) const = 0;
  
  virtual std::pair<float,float> pixel( const LocalPoint& p) const = 0;

  /// conversion taking also the angle from the track state
  virtual std::pair<float,float> pixel( const LocalPoint& p, const Topology::LocalTrackAngles &ltp ) const { 
    return pixel(p); 
  }
  
  virtual std::pair<float,float> pitch() const = 0;
  virtual int nrows() const = 0;
  virtual int ncolumns() const = 0;

  virtual float localX(const float mpX) const = 0;
  virtual float localY(const float mpY) const = 0;

/*  GF: These two need to be added! */
/*   virtual bool isItBigPixelInX(const int ixbin) const = 0;  */
/*   virtual bool isItBigPixelInY(const int iybin) const = 0;  */
  virtual bool containsBigPixelInX(const int& ixmin, const int& ixmax) const = 0;
  virtual bool containsBigPixelInY(const int& iymin, const int& iymax) const = 0;

  virtual bool isItEdgePixelInX (int ixbin) const = 0;
  virtual bool isItEdgePixelInY (int iybin) const = 0;
  virtual bool isItEdgePixel (int ixbin, int iybin) const = 0;

};

#endif
