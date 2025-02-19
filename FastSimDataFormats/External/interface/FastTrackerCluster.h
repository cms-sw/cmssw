#ifndef FastTrackerCluster_H
#define FastTrackerCluster_H

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/DetId/interface/DetId.h"

class FastTrackerCluster  
{
  
 public:
  
   FastTrackerCluster(): pos_(),err_(),id_(),simhitId_(),simtrackId_(),eeId_(),charge_() {}
  
  virtual ~FastTrackerCluster() {}
  
  FastTrackerCluster(const LocalPoint&, const LocalError&,
		   const DetId&,
		   const int simhitId, 
		   const int simtrackId, 
		   const uint32_t eeId, 
		   const float charge );
  
  
  const  LocalPoint& localPosition() const {return pos_;}
  const  LocalError& localPositionError() const{ return err_;}
  const  DetId&      id() const {return id_;}
  const  int&        simhitId() const {return simhitId_;}
  const  int&        simtrackId() const {return simtrackId_;}
  const  uint32_t&   eeId() const {return eeId_;}
  const  float&      charge() const {return charge_;}
  
  virtual FastTrackerCluster * clone() const {return new FastTrackerCluster( * this); }
  
 private:
  
  LocalPoint     pos_;
  LocalError     err_;
  DetId          id_;
  int const      simhitId_;
  int const      simtrackId_;
  uint32_t const eeId_;
  float const    charge_ ; 
  
};

// Comparison operators
inline bool operator<( const FastTrackerCluster& one, const FastTrackerCluster& other) {
  return ( one.simhitId() < other.simhitId() );
}


#endif
