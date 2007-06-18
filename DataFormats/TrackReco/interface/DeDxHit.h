#ifndef TrackDeDxHits_H
#define TrackDeDxHits_H
namespace reco {
/**
 * Class defining the dedx hits, i.e. track hits with only dedx need informations 
 */
class DeDxHit {
public:
  DeDxHit() {}
  DeDxHit(float ch,float dist,int detId):
  m_charge(ch),m_distance(dist),m_subDetId(detId) {}

  ///Return the angle and thick normalized, calibrated energy release
  float charge() const {return m_charge;}

  ///Return the distance of the hit from the interaction point
  float distance() const {return m_distance;}
  bool operator< (const DeDxHit & other) const {return m_charge < other.m_charge; }

private:
  float m_charge;
  float m_distance;
  int m_subDetId;
};


  typedef std::vector<DeDxHit> DeDxHitCollection;

}
#endif
