#include "DataFormats/Candidate/interface/Particle.h"
#include <memory>

const unsigned int reco::Particle::longLivedTag = 65536;

using namespace reco;

/// default constructor
Particle::Particle()
  : qx3_(0), pt_(0), eta_(0), phi_(0), mass_(0),
    vertex_(0, 0, 0), pdgId_(0), status_(0),
    p4Polar_(nullptr), p4Cartesian_(nullptr)
{
}
/// constructor from values
Particle::Particle( Charge q, const LorentzVector & p4, const Point & vertex,
      int pdgId, int status, bool integerCharge)
  : qx3_( q ), pt_( p4.pt() ), eta_( p4.eta() ), phi_( p4.phi() ), mass_( p4.mass() ),
    vertex_( vertex ), pdgId_( pdgId ), status_( status ),
    p4Polar_(nullptr), p4Cartesian_(nullptr)
{
    if ( integerCharge ) qx3_ *= 3;
}
/// constructor from values
Particle::Particle( Charge q, const PolarLorentzVector & p4, const Point & vertex,
      int pdgId, int status, bool integerCharge)
  : qx3_( q ), pt_( p4.pt() ), eta_( p4.eta() ), phi_( p4.phi() ), mass_( p4.mass() ),
    vertex_( vertex ), pdgId_( pdgId ), status_( status ),
    p4Polar_(nullptr), p4Cartesian_(nullptr)
{
    if ( integerCharge ) qx3_ *= 3;
}
// copy-ctor
Particle::Particle(const Particle& src)
  : qx3_(src.qx3_), pt_(src.pt_), eta_(src.eta_), phi_(src.phi_), mass_(src.mass_),
    vertex_(src.vertex_), pdgId_(src.pdgId_), status_(src.status_),
    p4Polar_(nullptr), p4Cartesian_(nullptr)
{
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
    other.p4Polar_.exchange(
            p4Polar_.exchange(other.p4Polar_, std::memory_order_acq_rel),
            std::memory_order_acq_rel);
    other.p4Cartesian_.exchange(
            p4Cartesian_.exchange(other.p4Cartesian_, std::memory_order_acq_rel),
            std::memory_order_acq_rel);
}
/// dtor
Particle::~Particle() {
    clearCache();
}

/// electric charge
int Particle::charge() const {
    return qx3_ / 3;
}
/// set electric charge
void Particle::setCharge( Charge q ) {
    qx3_ = q * 3;
}
/// electric charge
int Particle::threeCharge() const {
    return qx3_;
}
/// set electric charge
void Particle::setThreeCharge( Charge qx3 ) {
    qx3_ = qx3;
}
/// four-momentum Lorentz vector
const Particle::LorentzVector & Particle::p4() const {
    cacheCartesian();
    return (*p4Cartesian_.load(std::memory_order_acquire));
}
/// four-momentum Lorentz vector
const Particle::PolarLorentzVector & Particle::polarP4() const {
    cachePolar();
    return (*p4Polar_.load(std::memory_order_acquire));
}
/// spatial momentum vector
Particle::Vector Particle::momentum() const {
    cacheCartesian();
    return (*p4Cartesian_.load(std::memory_order_acquire)).Vect();
}
/// boost vector to boost a Lorentz vector
/// to the particle center of mass system
Particle::Vector Particle::boostToCM() const {
    cacheCartesian();
    return (*p4Cartesian_.load(std::memory_order_acquire)).BoostToCM();
}
/// magnitude of momentum vector
double Particle::p() const {
    cacheCartesian();
    return (*p4Cartesian_.load(std::memory_order_acquire)).P();
}
/// energy
double Particle::energy() const {
    cacheCartesian();
    return (*p4Cartesian_.load(std::memory_order_acquire)).E();
}
/// transverse energy
double Particle::et() const {
    cachePolar();
    return (*p4Polar_.load(std::memory_order_acquire)).Et();
}
/// mass
double Particle::mass() const {
    return mass_;
}
/// mass squared
double Particle::massSqr() const {
    return mass_ * mass_;
}
/// transverse mass
double Particle::mt() const {
    cachePolar();
    return (*p4Polar_.load(std::memory_order_acquire)).Mt();
}
/// transverse mass squared
double Particle::mtSqr() const {
    cachePolar();
    return (*p4Polar_.load(std::memory_order_acquire)).Mt2();
}
/// x coordinate of momentum vector
double Particle::px() const {
    cacheCartesian();
    return (*p4Cartesian_.load(std::memory_order_acquire)).Px();
}
/// y coordinate of momentum vector
double Particle::py() const {
    cacheCartesian();
    return (*p4Cartesian_.load(std::memory_order_acquire)).Py();
}
/// z coordinate of momentum vector
double Particle::pz() const {
    cacheCartesian();
    return (*p4Cartesian_.load(std::memory_order_acquire)).Pz();
}
/// transverse momentum
double Particle::pt() const {
    return pt_;
}
/// momentum azimuthal angle
double Particle::phi() const {
    return phi_;
}
/// momentum polar angle
double Particle::theta() const {
    cacheCartesian();
    return (*p4Cartesian_.load(std::memory_order_acquire)).Theta();
}
/// momentum pseudorapidity
double Particle::eta() const {
    return eta_;
}
/// repidity
double Particle::rapidity() const {
    cachePolar();
    return (*p4Polar_.load(std::memory_order_acquire)).Rapidity();
}
/// repidity
double Particle::y() const {
    return rapidity();
}

/// set 4-momentum
void Particle::setP4( const LorentzVector & p4 ) {
    // ensure that we have non-null pointers
    cacheCartesian();
    *p4Cartesian_.load(std::memory_order_acquire) = p4;
    // ensure that we have non-null pointers
    cachePolar();
    *p4Polar_.load(std::memory_order_acquire) = p4;
    auto const* p4Polar = p4Polar_.load(std::memory_order_acquire);
    pt_ = p4Polar->pt();
    eta_ = p4Polar->eta();
    phi_ = p4Polar->phi();
    mass_ = p4Polar->mass();
}
/// set 4-momentum
void Particle::setP4( const PolarLorentzVector & p4 ) {
    // ensure that we have non-null pointers
    cachePolar();
    *p4Polar_.load(std::memory_order_acquire) = p4;
    auto const* p4Polar = p4Polar_.load(std::memory_order_acquire);
    pt_ = p4Polar->pt();
    eta_ = p4Polar->eta();
    phi_ = p4Polar->phi();
    mass_ = p4Polar->mass();
    delete p4Cartesian_.exchange(nullptr, std::memory_order_acq_rel);
}
/// set particle mass
void Particle::setMass( double m ) {
    mass_ = m;
    clearCache();
}
void Particle::setPz( double pz ) {
    // ensure that we have non-null pointers
    cacheCartesian();
    (*p4Cartesian_.load(std::memory_order_acquire)).SetPz(pz);
    // ensure that we have non-null pointers
    cachePolar();
    (*p4Polar_.load(std::memory_order_acquire)) = (*p4Cartesian_.load(std::memory_order_acquire));
    auto const* p4Polar = p4Polar_.load(std::memory_order_acquire);
    pt_ = p4Polar->pt();
    eta_ = p4Polar->eta();
    phi_ = p4Polar->phi();
    mass_ = p4Polar->mass();
}

const Particle::Point & Particle::vertex() const {
    return vertex_;
}
/// x coordinate of vertex position
double Particle::vx() const {
    return vertex_.X();
}
/// y coordinate of vertex position
double Particle::vy() const {
    return vertex_.Y();
}
/// z coordinate of vertex position
double Particle::vz() const {
    return vertex_.Z();
}
/// set vertex
void Particle::setVertex( const Point & vertex ) {
    vertex_ = vertex;
}
/// PDG identifier
int Particle::pdgId() const {
    return pdgId_;
}
// set PDG identifier
void Particle::setPdgId( int pdgId ) {
    pdgId_ = pdgId;
}
/// status word
int Particle::status() const {
    return status_;
}
/// set status word
void Particle::setStatus( int status ) {
    status_ = status;
}
/// set long lived flag
void Particle::setLongLived() {
    status_ |= longLivedTag;
}
/// is long lived?
bool Particle::longLived() const {
    return status_ & longLivedTag;
}

/// set internal cache
void Particle::cachePolar() const {
    if(!p4Polar_.load(std::memory_order_acquire)) {
        std::unique_ptr<PolarLorentzVector> ptr{new PolarLorentzVector(pt_,eta_,phi_,mass_)};
        PolarLorentzVector* expect = nullptr;
        if(p4Polar_.compare_exchange_strong(expect, ptr.get(), std::memory_order_acq_rel)) {
            ptr.release();
        }
    }
}
/// set internal cache
void Particle::cacheCartesian() const {
    if(!p4Cartesian_.load(std::memory_order_acquire)) {
        cachePolar();
        std::unique_ptr<LorentzVector> ptr{new LorentzVector(*p4Polar_.load(std::memory_order_acquire))};
        LorentzVector* expected = nullptr;
        if( p4Cartesian_.compare_exchange_strong(expected, ptr.get(), std::memory_order_acq_rel) ) {
           ptr.release();
        }
    }
}
/// clear internal cache
void Particle::clearCache() const {
    delete p4Polar_.exchange(nullptr, std::memory_order_acq_rel);
    delete p4Cartesian_.exchange(nullptr, std::memory_order_acq_rel);
}
