#include "DataFormats/Candidate/interface/Particle.h"

const unsigned int reco::Particle::longLivedTag = 65536;

namespace reco {

/// default constructor
Particle::Particle()
  : qx3_(0), pt_(0), eta_(0), phi_(0), mass_(0),
    vertex_(0, 0, 0), pdgId_(0), status_(0)
{
    cachePolarFixed_.store(false, std::memory_order_release);
    cacheCartesianFixed_.store(false, std::memory_order_release);
}
/// constructor from values
Particle::Particle( Charge q, const LorentzVector & p4, const Point & vertex,
      int pdgId, int status, bool integerCharge)
  : qx3_( q ), pt_( p4.pt() ), eta_( p4.eta() ), phi_( p4.phi() ), mass_( p4.mass() ),
    vertex_( vertex ), pdgId_( pdgId ), status_( status )
{
    cachePolarFixed_.store(false, std::memory_order_release);
    cacheCartesianFixed_.store(false, std::memory_order_release);
    if ( integerCharge ) qx3_ *= 3;
}
/// constructor from values
Particle::Particle( Charge q, const PolarLorentzVector & p4, const Point & vertex,
      int pdgId, int status, bool integerCharge)
  : qx3_( q ), pt_( p4.pt() ), eta_( p4.eta() ), phi_( p4.phi() ), mass_( p4.mass() ),
    vertex_( vertex ), pdgId_( pdgId ), status_( status ) {
    cachePolarFixed_.store(false, std::memory_order_release);
    cacheCartesianFixed_.store(false, std::memory_order_release);
    if ( integerCharge ) qx3_ *= 3;
}
// copy-ctor
Particle::Particle(const Particle& src)
  : qx3_(src.qx3_), pt_(src.pt_), eta_(src.eta_), phi_(src.phi_), mass_(src.mass_),
    vertex_(src.vertex_), pdgId_(src.pdgId_), status_(src.status_) {
    cachePolarFixed_.store(false, std::memory_order_release);
    cacheCartesianFixed_.store(false, std::memory_order_release);
}
// copy assignment operator
Particle&
Particle::operator=(const Particle& rhs) {
    Particle temp(rhs);
    temp.swap(*this);
    return *this;
}
// public swap function
void Particle::swap(Particle& other) {
    std::swap(qx3_, other.qx3_);
    std::swap(pt_, other.pt_);
    std::swap(eta_, other.eta_);
    std::swap(phi_, other.phi_);
    std::swap(mass_, other.mass_);
    std::swap(vertex_, other.vertex_);
    std::swap(pdgId_, other.pdgId_);
    std::swap(status_, other.status_);
    other.cachePolarFixed_.exchange(cachePolarFixed_.exchange(other.cachePolarFixed_));
    other.cacheCartesianFixed_.exchange(cachePolarFixed_.exchange(other.cachePolarFixed_));
}

/// set 4-momentum
void Particle::setP4( const LorentzVector & p4 ) {
  p4Cartesian_ = p4;
  p4Polar_ = p4;
  pt_ = p4Polar_.pt();
  eta_ = p4Polar_.eta();
  phi_ = p4Polar_.phi();
  mass_ = p4Polar_.mass();
  cachePolarFixed_.store(true, std::memory_order_release);
  cacheCartesianFixed_.store(true, std::memory_order_release);
}
/// set 4-momentum
void Particle::setP4( const PolarLorentzVector & p4 ) {
  p4Polar_ = p4;
  pt_ = p4Polar_.pt();
  eta_ = p4Polar_.eta();
  phi_ = p4Polar_.phi();
  mass_ = p4Polar_.mass();
  cachePolarFixed_.store(true, std::memory_order_release);
  cacheCartesianFixed_.store(false, std::memory_order_release);
}
/// set particle mass
void Particle::setMass( double m ) {
  mass_ = m;
  clearCache();
}
void Particle::setPz( double pz ) {
  cacheCartesian();
  p4Cartesian_.SetPz(pz);
  p4Polar_ = p4Cartesian_;
  pt_ = p4Polar_.pt();
  eta_ = p4Polar_.eta();
  phi_ = p4Polar_.phi();
  mass_ = p4Polar_.mass();
}
} // end of reco namespace
