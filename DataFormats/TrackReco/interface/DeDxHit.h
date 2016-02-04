#ifndef TrackDeDxHits_H
#define TrackDeDxHits_H
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>

namespace reco {
/**
 * Class defining the dedx hits, i.e. track hits with only dedx need informations 
 */
class DeDxHit {
public:
  DeDxHit() {}
  DeDxHit(float ch,float dist,float len,DetId detId);

  ///Return the angle and thick normalized, calibrated energy release
  float charge() const {return m_charge;}

  ///Return the distance of the hit from the interaction point
  float distance() const {return m_distance;}
  
  ///Return the path length
  float pathLength() const {return m_pathLength;}
 
  /// Return the subdet
  int subDet() const {return (m_subDetId>>5)&0x7; }
  
  /// Return the plus/minus side for TEC/TID
  int subDetSide() const {return ((m_subDetId>>4)&0x1 )+ 1; }
  
  /// Return the layer/disk
  int layer() const {return m_subDetId & 0xF ; }

  /// Return the encoded layer + sub det id
  char subDetId() const {return m_subDetId; }

  bool operator< (const DeDxHit & other) const {return m_charge < other.m_charge; }

private:
  //Those data members should be "compressed" once usage 
  //of ROOT/reflex precision specifier will be available in CMSSW
  float m_charge;
  float m_distance;
  float m_pathLength;
  char m_subDetId;
  
};


  typedef std::vector<DeDxHit> DeDxHitCollection;

}
#endif
