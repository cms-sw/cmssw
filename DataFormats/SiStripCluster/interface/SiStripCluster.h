#ifndef DATAFORMATS_SISTRIPCLUSTER_H
#define DATAFORMATS_SISTRIPCLUSTER_H

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include <vector>
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class SiStripCluster  {
public:

  typedef std::vector<SiStripDigi>::const_iterator   SiStripDigiIter;
  typedef std::pair<SiStripDigiIter,SiStripDigiIter>   SiStripDigiRange;

  /** Construct from a range of digis that form a cluster and from 
   *  a DetID. The range is assumed to be non-empty.
   */
  
  SiStripCluster() : error_x( -99999.9 ) {}

  explicit SiStripCluster(const SiStripDigiRange& range);

  template<typename Iter>
  SiStripCluster(const uint16_t& firstStrip, 
		 Iter begin, Iter end ):
	 amplitudes_(begin,end), firstStrip_(firstStrip), 
  // ggiurgiu@fnal.gov, 01/05/12
  // Initialize the split cluster errors to un-physical values.
  // The CPE will check these errors and if they are not un-physical,
  // it will recognize the clusters as split and assign these (increased)
  // errors to the corresponding rechit.
  error_x(-99999.9){}

  /** The number of the first strip in the cluster
   */
  uint16_t firstStrip() const {return firstStrip_;}

  /** The amplitudes of the strips forming the cluster.
   *  The amplitudes are on consecutive strips; if a strip is missing
   *  the amplitude is set to zero.
   *  A strip may be missing in the middle of a cluster because of a
   *  clusterizer that accepts holes.
   *  A strip may also be missing anywhere in the cluster, including the 
   *  edge, to record a dead/noisy channel.
   *
   *  You can find the special meanings of values { 0, 254, 255} in section 3.4.1 of
   *  http://www.te.rl.ac.uk/esdg/cms-fed/firmware/Documents/FE_FPGA_Technical_Description.pdf
   */
  const std::vector<uint8_t>&  amplitudes() const {return amplitudes_;}

  /** The barycenter of the cluster, not corrected for Lorentz shift;
   *  should not be used as position estimate for tracking.
   */
  float barycenter() const;

  float getSplitClusterError () const    {  return error_x;  }
  void  setSplitClusterError ( float errx ) { error_x = errx; }


private:

  std::vector<uint8_t>   amplitudes_;

  uint16_t                firstStrip_;

  // ggiurgiu@fnal.gov, 01/05/12
  // Add cluster errors to be used by rechits from split clusters. 
  // A rechit from a split cluster has larger errors than rechits from normal clusters. 
  // However, when presented with a cluster, the CPE does not know if the cluster comes 
  // from a splitting procedure or not. That's why we have to instruct the CPE to use 
  // appropriate errors for split clusters.
  // To avoid increase of data size on disk,these new data members are set as transient in: 
  // DataFormats/SiStripCluster/src/classes_def.xml
  float error_x;
  
};

// Comparison operators
inline bool operator<( const SiStripCluster& one, const SiStripCluster& other) {
    return one.firstStrip() < other.firstStrip();
} 

inline bool operator<(const SiStripCluster& cluster, const uint16_t& firstStrip) {
  return cluster.firstStrip() < firstStrip;
} 

inline bool operator<(const uint16_t& firstStrip,const SiStripCluster& cluster) {
  return firstStrip < cluster.firstStrip();
} 
#endif // DATAFORMATS_SISTRIPCLUSTER_H
