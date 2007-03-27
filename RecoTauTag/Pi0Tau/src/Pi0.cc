#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTauTag/Pi0Tau/interface/Pi0.h"

using namespace reco;

Pi0::Pi0(){
  type_ = 0;
  energy_ = 0.0;
  position_ *= 0;
  momentum_ *= 0;
  sourceCandidates_.clear();
}


Pi0::Pi0(int type, double e, math::XYZPoint pos, math::XYZTLorentzVector mom, reco::PFCandidateRefVector &source_candidates){
  type_ = type;
  energy_ = e;
  position_ = pos;
  momentum_ = mom;
  sourceCandidates_ = source_candidates;
}
  

Pi0::Pi0(const Pi0& other){
  type_ = other.type();
  energy_ = other.energy();
  position_ = other.position();
  momentum_ = other.momentum();
  sourceCandidates_ = other.sourceCandidates();
}


math::XYZTLorentzVector Pi0::momentum(const math::XYZPoint &vtx) const {

  math::XYZTLorentzVector p4(0.0,0.0,0.0,0.0);

  double mag = momentum_.E();
  if(mag <= 0.0) return p4;

  if(mag > PI0MASS) mag = sqrt(mag*mag - PI0MASS*PI0MASS);

  math::XYZPoint p3(position_ - vtx);
  if(p3.R() == 0.0) return math::XYZTLorentzVector(0.0,0.0,0.0,0.0);

  p3 *= mag/p3.R();
  p4.SetCoordinates(p3.X(),p3.Y(),p3.Z(),momentum_.E());

  return p4;

}


std::ostream& reco::operator<<(std::ostream& out, 
			       const Pi0& pi0) {  
  if (!out) return out;  

  out << "type : " << pi0.type_
      << ", energy : "<< pi0.energy()
      << ", position = (" << pi0.position().X()
      << "," << pi0.position().Y()
      << "," << pi0.position().Z()
      << ")" 
      << std::endl;

  return out;
}
