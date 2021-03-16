#ifndef DataFormats_JetMatching_JetFlavour_H
#define DataFormats_JetMatching_JetFlavour_H

#include <vector>
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"

namespace reco {
  /**
 * JetFlavour class is meant to be used when the genEvent is dropped.
 * It can store by value the matching information about flavour and parton kinematics
 * The flavour definition and the corresponding parton information should be configured
 * in the producer.
 * The typedefs are taken from reco::Particle
 * */
  class JetFlavour {
  public:
    /// Lorentz vector
    typedef math::XYZTLorentzVector LorentzVector;
    /// point in the space
    typedef math::XYZPoint Point;
    /// lepton info struct
    struct Leptons {
      int electron;
      int muon;
      int tau;

      Leptons() : electron(0), muon(0), tau(0) {}
    };

    JetFlavour(void) {}
    JetFlavour(const LorentzVector& lv, const Point& pt, int fl, const Leptons& le)
        : m_partonMomentum(lv), m_partonVertex(pt), m_flavour(fl), m_leptons(le) {}
    JetFlavour(const LorentzVector& lv, const Point& pt, int fl)
        : m_partonMomentum(lv), m_partonVertex(pt), m_flavour(fl) {}

    const LorentzVector getLorentzVector() const { return m_partonMomentum; }
    const Point getPartonVertex() const { return m_partonVertex; }
    const int getFlavour() const { return m_flavour; }
    const Leptons getLeptons() const { return m_leptons; }

  private:
    LorentzVector m_partonMomentum;
    Point m_partonVertex;  // is it needed?
    int m_flavour;
    Leptons m_leptons;
  };

}  // namespace reco
#endif
