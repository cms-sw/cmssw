#ifndef TrackDeDxHits_H
#define TrackDeDxHits_H
namespace reco {
class DeDxHit {
public:
  DeDxHit() {}
  DeDxHit(float ch,float dist,int detId):
  m_charge(ch),m_distance(dist),m_subDetId(detId) {}

  //Return the angle and thick normalized, calibrated energy release
  float charge() const {return m_charge;}
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
