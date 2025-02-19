#ifndef HLTReco_TriggerObject_h
#define HLTReco_TriggerObject_h

/** \class trigger::TriggerObject
 *
 *  A single trigger object (e.g., an isolated muon, or MET)
 *  - described by its 4-momentum and physics type
 *
 *  $Date: 2010/10/14 23:00:36 $
 *  $Revision: 1.7 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Candidate/interface/Particle.h"
#include <cmath>
#include <vector>

namespace trigger
{

  /// Single trigger physics object (e.g., an isolated muon)
  class TriggerObject {

  /// data members - similar to DataFormats/Candidate/interface/Particle.h
  private:
    /// id or physics type (including electric charge) - similar to pdgId
    int id_;
    /// 4-momentum of physics object
    float pt_, eta_, phi_, mass_;

  /// methods
  public:
    /// constructors
    TriggerObject(): id_(), pt_(), eta_(), phi_(), mass_() { }
    TriggerObject(int id, float pt, float eta, float phi, float mass):
      id_(id), pt_(pt), eta_(eta), phi_(phi), mass_(mass) { }

    /// any type T object implementing the methods pt(), eta(), phi(), mass()
    template <typename T>
    TriggerObject(int id, const T& o):
    id_(id), pt_(o.pt()), eta_(o.eta()), phi_(o.phi()), mass_(o.mass()) { }
    /// ... and pdgId()
    template <typename T>
    TriggerObject(const T& o):
    id_(o.pdgId()), pt_(o.pt()), eta_(o.eta()), phi_(o.phi()), mass_(o.mass()) { }

    /// setters
    void setId  (int     id) {id_  =id;}
    void setPt  (float   pt) {pt_  =pt;}
    void setEta (float  eta) {eta_ =eta;}
    void setPhi (float  phi) {phi_ =phi;}
    void setMass(float mass) {mass_=mass;}

    /// getters
    int   id() const {return id_;}
    float pt() const {return pt_;}
    float eta() const {return eta_;}
    float phi() const {return phi_;}
    float mass() const {return mass_;}

    float px() const {return pt_*std::cos(phi_);}
    float py() const {return pt_*std::sin(phi_);}
    float pz() const {return pt_*std::sinh(eta_);}
    float p () const {return pt_*std::cosh(eta_);}
    float energy() const {return std::sqrt(std::pow(mass_,2)+std::pow(p(),2));}
    // et = energy/cosh(eta)
    float et() const {return std::sqrt(std::pow(mass_/std::cosh(eta_),2)+std::pow(pt_,2));}

    reco::Particle particle(reco::Particle::Charge q=0, 
      const reco::Particle::Point & vertex = reco::Particle::Point(0,0,0),
      int status=0, bool integerCharge=true) const {
      return reco::Particle(q,
        reco::Particle::LorentzVector(px(),py(),pz(),energy()),
        vertex,id(),status,integerCharge);
    }

  };


  /// collection of trigger physics objects (e.g., all isolated muons)
  typedef std::vector<TriggerObject> TriggerObjectCollection;

}

#endif
