#ifndef DATAFORMATS_SISTRIPCLUSTER_H
#define DATAFORMATS_SISTRIPCLUSTER_H

//#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include <vector>

class SiStripCluster  /*: public edm::DoNotSortUponInsertion*/ {
public:

  typedef std::vector<SiStripDigi>::const_iterator   SiStripDigiIter;
  typedef std::pair<SiStripDigiIter,SiStripDigiIter>   SiStripDigiRange;

  /** Construct from a range of digis that form a cluster and from 
   *  a DetID. The range is assumed to be non-empty.
   */
  
  SiStripCluster() : detId_(0) {}

  SiStripCluster( uint32_t detid, const SiStripDigiRange& range);

  SiStripCluster(const uint32_t& detid, 
		 const uint16_t& firstStrip, 
		 std::vector<uint16_t>::const_iterator begin, 
		 std::vector<uint16_t>::const_iterator end );

  /** The number of the first strip in the cluster
   */
  uint16_t firstStrip() const {return firstStrip_;}

  /** The geographical ID of the corresponding DetUnit, 
   *  to be used for transformations to local and to global reference 
   *  frames etc.
   */
  uint32_t geographicalId() const {return detId_;}

  /** The amplitudes of the strips forming the cluster.
   *  The amplitudes are on consecutive strips; if a strip is missing
   *  the amplitude is set to zero.
   *  A strip may be missing in the middle of a cluster because of a
   *  clusterizer that accepts holes.
   *  A strip may also be missing anywhere in the cluster, including the 
   *  edge, to record a dead/noisy channel.
   */
  const std::vector<uint8_t>&  amplitudes() const {return amplitudes_;}

  /** The barycenter of the cluster, not corrected for Lorentz shift;
   *  should not be used as position estimate for tracking.
   */
  float barycenter() const;

private:

  uint32_t                detId_;
  uint16_t                firstStrip_;
  std::vector<uint8_t>   amplitudes_;
};

// Comparison operators
inline bool operator<( const SiStripCluster& one, const SiStripCluster& other) {
  if(one.geographicalId() == other.geographicalId()) {
    return one.firstStrip() < other.firstStrip();
  }
  return one.geographicalId() < other.geographicalId();
} 

inline bool operator<(const SiStripCluster& cluster, const uint32_t& detid) {
  return cluster.geographicalId() < detid;
} 

inline bool operator<(const uint32_t& detid,const SiStripCluster& cluster) {
  return detid < cluster.geographicalId();
} 

inline bool operator<(const SiStripCluster& cluster, const uint16_t& firstStrip) {
  return cluster.firstStrip() < firstStrip;
} 

inline bool operator<(const uint16_t& firstStrip,const SiStripCluster& cluster) {
  return firstStrip < cluster.firstStrip();
} 
#endif // DATAFORMATS_SISTRIPCLUSTER_H
