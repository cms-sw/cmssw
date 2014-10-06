#ifndef DATAFORMATS_SISTRIPCLUSTER_H
#define DATAFORMATS_SISTRIPCLUSTER_H

#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include <vector>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<cassert>

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
	 size_(end-begin), firstStrip_(firstStrip),
  // ggiurgiu@fnal.gov, 01/05/12
  // Initialize the split cluster errors to un-physical values.
  // The CPE will check these errors and if they are not un-physical,
  // it will recognize the clusters as split and assign these (increased)
  // errors to the corresponding rechit.
  error_x(-99999.9){
    assert(size_<=MAX_SIZE);
    std::copy(begin,end,amplitudes_);
  }

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
  const SiStripCluster &  amplitudes() const {return *this;}

  /** The barycenter of the cluster, not corrected for Lorentz shift;
   *  should not be used as position estimate for tracking.
   */
  float barycenter() const;

  float getSplitClusterError () const    {  return error_x;  }
  void  setSplitClusterError ( float errx ) { error_x = errx; }

  enum { MAX_SIZE=16};

  uint8_t const & front() const { return amplitudes_[0];}
  uint8_t const * begin() const { return amplitudes_;}
  uint8_t const * end() const	{ return amplitudes_+size_;}
  uint16_t        size() const  { return size_;}
  uint8_t operator[](unsigned int i) const { return amplitudes_[i];}
  uint8_t & operator[](unsigned int i) { return amplitudes_[i];}

private:

  uint8_t   amplitudes_[MAX_SIZE];

  uint16_t                size_;
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
