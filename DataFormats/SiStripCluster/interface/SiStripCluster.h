#ifndef DATAFORMATS_SISTRIPCLUSTER_H
#define DATAFORMATS_SISTRIPCLUSTER_H

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"

class StripDigi;

class SiStripCluster {
public:

  typedef std::vector<StripDigi>::const_iterator   StripDigiIter;
  typedef std::pair<StripDigiIter,StripDigiIter>   StripDigiRange;

  /** Construct from a range of digis that form a cluster and from 
   *  a DetID. The range is assumed to be non-empty.
   */
  
  SiStripCluster() : detId_(0) {}

  SiStripCluster( unsigned int detid, const StripDigiRange& range);

  /** The number of the first strip in the cluster
   */
  short firstStrip() const {return firstStrip_;}

  /** The geographical ID of the corresponding DetUnit, 
   *  to be used for transformations to local and to global reference 
   *  frames etc.
   */
  cms::DetId geographicalId() const {return detId_;}

  /** The amplitudes of the strips forming the cluster.
   *  The amplitudes are on consecutive strips; if a strip is missing
   *  the amplitude is set to zero.
   *  A strip may be missing in the middle of a cluster because of a
   *  clusterizer that accepts holes.
   *  A strip may also be missing anywhere in the cluster, including the 
   *  edge, to record a dead/noisy channel.
   */
  const std::vector<short>&  amplitudes() const {return amplitudes_;}

  /** The barycenter of the cluster, not corrected for Lorentz shift;
   *  should not be used as position estimate for tracking.
   */
  float barycenter() const {return barycenter_;}
  float barycenter_error() const {return barycenter_error_;}

private:

  cms::DetId           detId_;
  short                firstStrip_;
  std::vector<short>   amplitudes_;
  float                barycenter_;
  float                barycenter_error_;

};

// Comparison operators
inline bool operator<( const SiStripCluster& one, const SiStripCluster& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else if ( one.geographicalId().rawId() > other.geographicalId().rawId() ) {
    return false;
  } else {
    if ( one.firstStrip() <= other.firstStrip() ) {
    return true;
  } else {
    return false;
    }
  }
}

#endif // DATAFORMATS_SISTRIPCLUSTER_H
