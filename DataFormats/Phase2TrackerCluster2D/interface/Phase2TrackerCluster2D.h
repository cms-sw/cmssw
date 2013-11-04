#ifndef DataFormats_Phase2TrackerCluster2D_h
#define DataFormats_Phase2TrackerCluster2D_h

#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#ifndef CMS_NOCXX11
#include <cstdint>
#else
#include "boost/cstdint.hpp"
#endif

#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include <vector>
#include <algorithm>

class Phase2TrackerCluster2D {

  public:
  
  Phase2TrackerCluster2D() : referenceDigi_(), offset_() {} 

  Phase2TrackerCluster2D( const std::vector<Phase2TrackerDigi>& digis ) {
    // assumes a non-empty input
    assert(digis.size());
    // take the first digi as a reference
    // we could avoid the search for the minimum if the vector was sorted.
    std::vector<Phase2TrackerDigi>::const_iterator firstdigi = std::min_element(digis.begin(),digis.end());
    referenceDigi_ = * firstdigi;
    unsigned int refx = referenceDigi_.row();
    unsigned int refy = referenceDigi_.column();
    for(std::vector<Phase2TrackerDigi>::const_iterator it = digis.begin();it<digis.end();++it) {
      // skip the first digi (no need to store the (0,0) offset)
      if (it==firstdigi) continue;
      // computes and saves the offsetx
      offset_.push_back(it->row()-refx);
      // computes and saves the offsety
      offset_.push_back(it->column()-refy);
    }
  }

  Phase2TrackerCluster2D( const Phase2TrackerDigi& referencePoint, const std::vector<uint8_t>& offsets ) {
    // pretty simple, that one
    referenceDigi_ = referencePoint;
    offset_ = offsets;
  }

  Phase2TrackerCluster2D( const Phase2TrackerDigi& referencePoint, int len, uint8_t const *offsetr, uint8_t const *offsetc ) {
    // just trust everything
    referenceDigi_ = referencePoint;
    for(int i=0;i<len;++i) {
      offset_.push_back(offsetr[i]);
      offset_.push_back(offsetc[i]);
    }
  } 

  void addHit(const Phase2TrackerDigi& hit) {
    offset_.push_back(hit.row()-referenceDigi_.row());
    offset_.push_back(hit.column()-referenceDigi_.column());
  }

  // fast access to inner members
  const Phase2TrackerDigi & referencePoint() const { return referenceDigi_; }

  const std::vector<uint8_t> & offsets() const { return offset_; }

  // Analog linear average position (barycenter) 
  float row() const {
    int sum = 0.;
    for(std::vector<uint8_t>::const_iterator it=offset_.begin();it<offset_.end();it+=2) {
      sum += *it;
    }
    return referenceDigi_.row()+sum/(offset_.size()/2+1.);
  }
  
  float column() const {
    int sum = 0.;
    for(std::vector<uint8_t>::const_iterator it=offset_.begin()+1;it<offset_.end();it+=2) {
      sum += *it;
    }
    return referenceDigi_.column()+sum/(offset_.size()/2+1.);
  }
  
  // Return number of digis.
  int size() const { return offset_.size()+1; }
  
  // Return cluster dimension in the x direction.
  int sizeX() const {
    int mini = 0;
    int maxi = 0;
    for(std::vector<uint8_t>::const_iterator it=offset_.begin();it<offset_.end();it+=2) {
      if (mini>*it) mini = *it;
      if (maxi<*it) maxi = *it;
    }
    return maxi-mini+1;
  }
  
  // Return cluster dimension in the y direction.
  int sizeY() const {
    int mini = 0;
    int maxi = 0;
    for(std::vector<uint8_t>::const_iterator it=offset_.begin();it<offset_.end();it+=2) {
      if (mini>*it) mini = *it;
      if (maxi<*it) maxi = *it;
    }
    return maxi-mini+1;
  }
  
  Phase2TrackerDigi digi(unsigned int i) const {
    if (i==0) return referenceDigi_;
    return Phase2TrackerDigi(referenceDigi_.row()+offset_[2*(i-1)], referenceDigi_.column()+offset_[2*(i-1)+1]);
  }
  
private:
  
  Phase2TrackerDigi referenceDigi_;
  std::vector<uint8_t>  offset_;
  
};

// Comparison operator
inline bool operator<( const Phase2TrackerCluster2D& one, const Phase2TrackerCluster2D& other) {
  if (one.row() < other.row()) return true;
  else if (one.row() > other.row()) return false;
  else if (one.column() < other.column()) return true;
  else return false;
}

#endif 
