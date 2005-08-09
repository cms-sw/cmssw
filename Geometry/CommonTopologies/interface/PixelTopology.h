#ifndef Geometry_CommonTopologies_PixelTopology_H
#define Geometry_CommonTopologies_PixelTopology_H

#include "Geometry/CommonTopologies/interface/Topology.h"


class Topology;


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
  
  virtual std::pair<float,float> pitch() const = 0;
  virtual int nrows() const = 0;
  virtual int ncolumns() const = 0;
  
};

#endif
