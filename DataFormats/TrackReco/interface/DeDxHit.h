#ifndef TrackDeDxHits_H
#define TrackDeDxHits_H
#include <vector>

namespace reco
{
/**
 * Class defining the dedx hits, i.e. track hits with only dedx need informations
 */
class DeDxHit
{

public:
    DeDxHit() {}

    DeDxHit(float ch, float mom, float len, uint32_t rawDetId):
      m_charge(ch),
      m_momentum(mom),
      m_pathLength(len),
      m_rawDetId(rawDetId){
    }

    /// Return the angle and thick normalized, calibrated energy release
    float charge() const {
        return m_charge;
    }

    /// Return the momentum of the trajectory at the interaction point
    float momentum() const {
        return m_momentum;
    }

    /// Return the path length
    float pathLength() const {
        return m_pathLength;
    }

    /// Return the rawDetId
    uint32_t rawDetId() const {
        return m_rawDetId;
    }

    bool operator< (const DeDxHit &other) const {
        return m_charge < other.m_charge;
    }

private:
    // Those data members should be "compressed" once usage
    // of ROOT/reflex precision specifier will be available in CMSSW
    float m_charge;
    float m_momentum;
    float m_pathLength;
    uint32_t m_rawDetId;
};

typedef std::vector<DeDxHit> DeDxHitCollection;

} // namespace reco
#endif

