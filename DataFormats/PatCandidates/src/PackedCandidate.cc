#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/PatCandidates/interface/libminifloat.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "DataFormats/PatCandidates/interface/liblogintpack.h"
using namespace logintpack;

void pat::PackedCandidate::pack(bool unpackAfterwards) {
    packedPt_  =  MiniFloatConverter::float32to16(p4_.Pt());
    packedEta_ =  int16_t(std::round(p4_.Eta()/6.0f*std::numeric_limits<int16_t>::max()));
    packedPhi_ =  int16_t(std::round(p4_.Phi()/3.2f*std::numeric_limits<int16_t>::max()));
    packedM_   =  MiniFloatConverter::float32to16(p4_.M());
    if (unpackAfterwards) unpack(); // force the values to match with the packed ones
}

void pat::PackedCandidate::packVtx(bool unpackAfterwards) {
    Point pv = pvRef_.isNonnull() ? pvRef_->position() : Point();
    float dxPV = vertex_.X() - pv.X(), dyPV = vertex_.Y() - pv.Y(); //, rPV = std::hypot(dxPV, dyPV);
    float s = std::sin(float(p4_.Phi())+dphi_), c = std::cos(float(p4_.Phi()+dphi_)); // not the fastest option, but we're in reduced precision already, so let's avoid more roundoffs
    dxy_  = - dxPV * s + dyPV * c;    
    // if we want to go back to the full x,y,z we need to store also
    // float dl = dxPV * c + dyPV * s; 
    // float xRec = - dxy_ * s + dl * c, yRec = dxy_ * c + dl * s;
    float pzpt = p4_.Pz()/p4_.Pt();
    dz_ = vertex_.Z() - pv.Z() - (dxPV*c + dyPV*s) * pzpt;
    packedDxy_ = MiniFloatConverter::float32to16(dxy_*100);
    packedDz_   = pvRef_.isNonnull() ? MiniFloatConverter::float32to16(dz_*100) : int16_t(std::round(dz_/40.f*std::numeric_limits<int16_t>::max()));
    packedDPhi_ =  int16_t(std::round(dphi_/3.2f*std::numeric_limits<int16_t>::max()));
    packedCovarianceDxyDxy_ = MiniFloatConverter::float32to16(dxydxy_*10000.);
    packedCovarianceDxyDz_ = MiniFloatConverter::float32to16(dxydz_*10000.);
    packedCovarianceDzDz_ = MiniFloatConverter::float32to16(dzdz_*10000.);
//    packedCovarianceDxyDxy_ = pack8log(dxydxy_,-15,-1); // MiniFloatConverter::float32to16(dxydxy_*10000.);
//    packedCovarianceDxyDz_ = pack8log(dxydz_,-20,-1); //MiniFloatConverter::float32to16(dxydz_*10000.);
//  packedCovarianceDzDz_ = pack8log(dzdz_,-13,-1); //MiniFloatConverter::float32to16(dzdz_*10000.);
    packedCovarianceDphiDxy_ = pack8log(dphidxy_,-17,-4); // MiniFloatConverter::float32to16(dphidxy_*10000.);
    packedCovarianceDlambdaDz_ = pack8log(dlambdadz_,-17,-4); // MiniFloatConverter::float32to16(dlambdadz_*10000.);
    packedCovarianceDptDpt_ = pack8log(dptdpt_,-15,5,32);
    packedCovarianceDetaDeta_ = pack8log(detadeta_,-20,0,32);
    packedCovarianceDphiDphi_ = pack8log(dphidphi_,-15,5,32);
    if (unpackAfterwards) unpackVtx();
}

void pat::PackedCandidate::unpack() const {
    p4_ = PolarLorentzVector(MiniFloatConverter::float16to32(packedPt_),
                             int16_t(packedEta_)*6.0f/std::numeric_limits<int16_t>::max(),
                             int16_t(packedPhi_)*3.2f/std::numeric_limits<int16_t>::max(),
                             MiniFloatConverter::float16to32(packedM_));
    p4c_ = p4_;
    unpacked_ = true;
}
void pat::PackedCandidate::unpackVtx() const {
    dphi_ = int16_t(packedDPhi_)*3.2f/std::numeric_limits<int16_t>::max(),
    dxy_ = MiniFloatConverter::float16to32(packedDxy_)/100.;
    dz_   = pvRef_.isNonnull() ? MiniFloatConverter::float16to32(packedDz_)/100. : int16_t(packedDz_)*40.f/std::numeric_limits<int16_t>::max();
    Point pv = pvRef_.isNonnull() ? pvRef_->position() : Point();
    float phi = p4_.Phi()+dphi_, s = std::sin(phi), c = std::cos(phi);
    vertex_ = Point(pv.X() - dxy_ * s,
                    pv.Y() + dxy_ * c,
                    pv.Z() + dz_ ); // for our choice of using the PCA to the PV, by definition the remaining term -(dx*cos(phi) + dy*sin(phi))*(pz/pt) is zero
//  dxydxy_ = unpack8log(packedCovarianceDxyDxy_,-15,-1);
//  dxydz_ = unpack8log(packedCovarianceDxyDz_,-20,-1);
//  dzdz_ = unpack8log(packedCovarianceDzDz_,-13,-1);
    dphidxy_ = unpack8log(packedCovarianceDphiDxy_,-17,-4);
    dlambdadz_ = unpack8log(packedCovarianceDlambdaDz_,-17,-4);
    dptdpt_ = unpack8log(packedCovarianceDptDpt_,-15,5,32);
    detadeta_ = unpack8log(packedCovarianceDetaDeta_,-20,0,32);
    dphidphi_ = unpack8log(packedCovarianceDphiDphi_,-15,5,32);

  dxydxy_ = MiniFloatConverter::float16to32(packedCovarianceDxyDxy_)/10000.;
    dxydz_ =MiniFloatConverter::float16to32(packedCovarianceDxyDz_)/10000.;
    dzdz_ =MiniFloatConverter::float16to32(packedCovarianceDzDz_)/10000.;
/*  dphidxy_ = MiniFloatConverter::float16to32(packedCovarianceDphiDxy_)/10000.;
    dlambdadz_ =MiniFloatConverter::float16to32(packedCovarianceDlambdaDz_)/10000.;
*/
    unpackedVtx_ = true;
}

pat::PackedCandidate::~PackedCandidate() { }


float pat::PackedCandidate::dxy(const Point &p) const {
	maybeUnpackBoth();
	return -(vertex_.X()-p.X()) * std::sin(float(p4_.Phi())) + (vertex_.Y()-p.Y()) * std::cos(float(p4_.Phi()));
}
float pat::PackedCandidate::dz(const Point &p) const {
    maybeUnpackBoth();
    return (vertex_.Z()-p.Z())  - ((vertex_.X()-p.X()) * std::cos(float(p4_.Phi())) + (vertex_.Y()-p.Y()) * std::sin(float(p4_.Phi()))) * p4_.Pz()/p4_.Pt();
}

reco::Track pat::PackedCandidate::pseudoTrack() const {
    maybeUnpackBoth();
    reco::TrackBase::CovarianceMatrix m;
//    m(0,0)=0.5e-4/pt()/pt(); //TODO: tune
//    m(1,1)=6e-6; //TODO: tune 
//    m(2,2)=1.5e-5/pt()/pt(); //TODO: tune
    m(0,0)=dptdpt_/pt()/pt(); //TODO: tune
    m(1,1)=detadeta_; //TODO: tune 
    m(2,2)=dphidphi_/pt()/pt(); //TODO: tune
    m(2,3)=dphidxy_;
    m(3,2)=dphidxy_;
    m(4,1)=dlambdadz_;
    m(1,4)=dlambdadz_;
    m(3,3)=dxydxy_;
    m(3,4)=dxydz_;
    m(4,3)=dxydz_;
    m(4,4)=dzdz_;
    math::RhoEtaPhiVector p3(p4_.pt(),p4_.eta(),phiAtVtx());
    int numberOfPixelHits_ = packedHits_ & 0x7 ;
    int numberOfHits_ = (packedHits_>>3) + numberOfPixelHits_;

    int ndof = numberOfHits_+numberOfPixelHits_-5;
    reco::HitPattern hp, hpExpIn;
    int i=0;
    LostInnerHits innerLost = lostInnerHits();
    switch (innerLost) {
        case validHitInFirstPixelBarrelLayer:
            hp.set(PXBDetId(1,0,0), TrackingRecHit::valid, i); 
            i++; 
            break;
        case noLostInnerHits:
            break;
        case oneLostInnerHit:
            hpExpIn.set(PXBDetId(1,0,0), TrackingRecHit::missing, 0);
            break;
        case moreLostInnerHits:
            hpExpIn.set(PXBDetId(1,0,0), TrackingRecHit::missing, 0);
            hpExpIn.set(PXBDetId(2,0,0), TrackingRecHit::missing, 1);
            break;
    };
    for(;i<numberOfPixelHits_; i++) {
       hp.set( PXBDetId(i>1?3:2,0,0), TrackingRecHit::valid, i); 
    }
    for(;i<numberOfHits_;i++) {
	   hp.set( TIBDetId(1,0,0,1,1,0), TrackingRecHit::valid, i); 
     }
    reco::Track tk(normalizedChi2_*ndof,ndof,vertex_,math::XYZVector(p3.x(),p3.y(),p3.z()),charge(),m,reco::TrackBase::undefAlgorithm,reco::TrackBase::loose);
    tk.setHitPattern(hp);
    tk.setTrackerExpectedHitsInner(hpExpIn);
    if (trackHighPurity()) tk.setQuality(reco::TrackBase::highPurity);
    return tk;
}

//// Everything below is just trivial implementations of reco::Candidate methods

pat::PackedCandidate::const_iterator pat::PackedCandidate::begin() const { 
  return const_iterator( new const_iterator_imp_specific ); 
}

pat::PackedCandidate::const_iterator pat::PackedCandidate::end() const { 
  return  const_iterator( new const_iterator_imp_specific ); 
}

pat::PackedCandidate::iterator pat::PackedCandidate::begin() { 
  return iterator( new iterator_imp_specific ); 
}

pat::PackedCandidate::iterator pat::PackedCandidate::end() { 
  return iterator( new iterator_imp_specific ); 
}

const reco::CandidateBaseRef & pat::PackedCandidate::masterClone() const {
  throw cms::Exception("Invalid Reference")
    << "this Candidate has no master clone reference."
    << "Can't call masterClone() method.\n";
}

bool pat::PackedCandidate::hasMasterClone() const {
  return false;
}

bool pat::PackedCandidate::hasMasterClonePtr() const {
  return false;
}


const reco::CandidatePtr & pat::PackedCandidate::masterClonePtr() const {
  throw cms::Exception("Invalid Reference")
    << "this Candidate has no master clone ptr."
    << "Can't call masterClonePtr() method.\n";
}

size_t pat::PackedCandidate::numberOfDaughters() const { 
  return 0; 
}

size_t pat::PackedCandidate::numberOfMothers() const { 
  return 0; 
}

bool pat::PackedCandidate::overlap( const reco::Candidate & o ) const { 
  return  p4() == o.p4() && vertex() == o.vertex() && charge() == o.charge();
//  return  p4() == o.p4() && charge() == o.charge();
}

const reco::Candidate * pat::PackedCandidate::daughter( size_type ) const {
  return 0;
}

const reco::Candidate * pat::PackedCandidate::mother( size_type ) const {
  return 0;
}

const reco::Candidate * pat::PackedCandidate::daughter(const std::string&) const {
  throw edm::Exception(edm::errors::UnimplementedFeature)
    << "This Candidate type does not implement daughter(std::string). "
    << "Please use CompositeCandidate or NamedCompositeCandidate.\n";
}

reco::Candidate * pat::PackedCandidate::daughter(const std::string&) {
  throw edm::Exception(edm::errors::UnimplementedFeature)
    << "This Candidate type does not implement daughter(std::string). "
    << "Please use CompositeCandidate or NamedCompositeCandidate.\n";
}



reco::Candidate * pat::PackedCandidate::daughter( size_type ) {
  return 0;
}

double pat::PackedCandidate::vertexChi2() const {
  return 0;
}

double pat::PackedCandidate::vertexNdof() const {
  return 0;
}

double pat::PackedCandidate::vertexNormalizedChi2() const {
  return 0;
}

double pat::PackedCandidate::vertexCovariance(int i, int j) const {
  throw edm::Exception(edm::errors::UnimplementedFeature)
    << "reco::ConcreteCandidate does not implement vertex covariant matrix.\n";
}

void pat::PackedCandidate::fillVertexCovariance(CovarianceMatrix & err) const {
  throw edm::Exception(edm::errors::UnimplementedFeature)
    << "reco::ConcreteCandidate does not implement vertex covariant matrix.\n";
}


bool pat::PackedCandidate::longLived() const {return false;}

bool pat::PackedCandidate::massConstraint() const {return false;}




