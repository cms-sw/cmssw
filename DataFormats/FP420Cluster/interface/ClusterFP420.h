#ifndef ClusterFP420_h
#define ClusterFP420_h

#include <vector>
class HDigiFP420;

class ClusterFP420 {
public:

  typedef std::vector<HDigiFP420>::const_iterator   HDigiFP420Iter;
  typedef std::pair<HDigiFP420Iter,HDigiFP420Iter>   HDigiFP420Range;

  //ClusterFP420() : detId_(0) , xytype_(0) {}
  ClusterFP420() : detId_(0)  {}

  //The range is assumed to be non-empty.
  ClusterFP420( unsigned int, unsigned int, const HDigiFP420Range&, float&, float&);
  //  ClusterFP420( unsigned int detid, const HDigiFP420Range& range, float& cog, float& err);// work also

  // number of the first strip in the cluster
  short firstStrip() const {return firstStrip_;}

  //global ID of the corresponding DetUnit --> iu index
  unsigned int globalId() const {return detId_;}

  // since xytype=2 all the time, no sense to record it into collection, so do comment the next line:
  //unsigned int globalType() const {return xytype_;}

  const std::vector<short>&  amplitudes() const {return amplitudes_;}

  float barycenter() const {return barycenter_;}
  float barycerror() const {return barycerror_;}

  float barycenterW() const {return barycenterW_;}
  float barycerrorW() const {return barycerrorW_;}

private:

  unsigned int           detId_;
  //  unsigned int           xytype_;
  short                firstStrip_;
  std::vector<short>   amplitudes_;
  float                barycenter_;
  float                barycerror_;
  float                barycenterW_;
  float                barycerrorW_;


};

// Comparison operators
inline bool operator<( const ClusterFP420& one, const ClusterFP420& other) {
  if(one.globalId() == other.globalId()) {
    return one.firstStrip() < other.firstStrip();
  }
  return one.globalId() < other.globalId();
} 
#endif 
