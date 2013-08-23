#include "DataFormats/RecoCandidate/interface/IsoDepositVetos.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"

#include <cmath>
#include <iostream>

using namespace reco::isodeposit;

bool ConeVeto::veto(double eta, double phi, float value) const {
  return ( vetoDir_.deltaR2(Direction(eta,phi)) < dR2_ );
}
void ConeVeto::centerOn(double eta, double phi) { 
	vetoDir_ = Direction(eta,phi);
}
///////////////////////////////////////////////////////////////////////////

bool ThresholdVeto::veto(double eta, double phi, float value) const {
    return (value <= threshold_);
}
void ThresholdVeto::centerOn(double eta, double phi) { }

///////////////////////////////////////////////////////////////////////////

bool ThresholdVetoFromTransverse::veto(double eta, double phi, float value) const {
  return ( value/sin(2*atan(exp(-eta)))  <= threshold_);  // convert Et to E
}
void ThresholdVetoFromTransverse::centerOn(double eta, double phi) { }

///////////////////////////////////////////////////////////////////////////

bool AbsThresholdVeto::veto(double eta, double phi, float value) const {
    return ( fabs(value) <= threshold_);
}
void AbsThresholdVeto::centerOn(double eta, double phi) { }

///////////////////////////////////////////////////////////////////////////

bool AbsThresholdVetoFromTransverse::veto(double eta, double phi, float value) const {
  return ( fabs(value/sin(2*atan(exp(-eta))))  <= threshold_);  // convert Et to E
}
void AbsThresholdVetoFromTransverse::centerOn(double eta, double phi) { }

///////////////////////////////////////////////////////////////////////////

bool ConeThresholdVeto::veto(double eta, double phi, float value) const {
  return (value <= threshold_) || ( vetoDir_.deltaR2(Direction(eta,phi)) < dR2_ );
}
void ConeThresholdVeto::centerOn(double eta, double phi) { 
	vetoDir_ = Direction(eta,phi);
}

///////////////////////////////////////////////////////////////////////////

AngleConeVeto::AngleConeVeto(const math::XYZVectorD& dir, double angle) : vetoDir_(dir.Unit()), cosTheta_(cos(angle)) {
}
AngleConeVeto::AngleConeVeto(Direction dir, double angle) : vetoDir_(0,0,1), cosTheta_(cos(angle)) {
    vetoDir_ = math::RhoEtaPhiVectorD(1, dir.eta(), dir.phi()).Unit(); 
}
bool AngleConeVeto::veto(double eta, double phi, float value) const {
    math::RhoEtaPhiVectorD tmp(1, eta, phi); 
    return ( vetoDir_.Dot(tmp.Unit()) > cosTheta_ );
}
void AngleConeVeto::centerOn(double eta, double phi) { 
	vetoDir_ = math::RhoEtaPhiVectorD(1, eta, phi).Unit(); 
}

///////////////////////////////////////////////////////////////////////////

AngleCone::AngleCone(const math::XYZVectorD& dir, double angle) : coneDir_(dir.Unit()), cosTheta_(cos(angle)) {
}
AngleCone::AngleCone(Direction dir, double angle) : coneDir_(0,0,1), cosTheta_(cos(angle)) {
    coneDir_ = math::RhoEtaPhiVectorD(1, dir.eta(), dir.phi()).Unit(); 
}
bool AngleCone::veto(double eta, double phi, float value) const {
    math::RhoEtaPhiVectorD tmp(1, eta, phi); 
    return ( coneDir_.Dot(tmp.Unit()) < cosTheta_ );
}
void AngleCone::centerOn(double eta, double phi) { 
	coneDir_ = math::RhoEtaPhiVectorD(1, eta, phi).Unit(); 
}

///////////////////////////////////////////////////////////////////////////
		
RectangularEtaPhiVeto::RectangularEtaPhiVeto(const math::XYZVectorD& dir, double etaMin, double etaMax, double phiMin, double phiMax) :
	vetoDir_(dir.eta(),dir.phi()), etaMin_(etaMin), etaMax_(etaMax), phiMin_(phiMin), phiMax_(phiMax) {
}

RectangularEtaPhiVeto::RectangularEtaPhiVeto(Direction dir, double etaMin, double etaMax, double phiMin, double phiMax) :
	vetoDir_(dir.eta(),dir.phi()), etaMin_(etaMin), etaMax_(etaMax), phiMin_(phiMin), phiMax_(phiMax) {
}

bool RectangularEtaPhiVeto::veto(double eta, double phi, float value) const  {
	//vetoDir_.phi() is already [0,2*M_PI], make sure the vetoDir phi is 
	//also assuming that the etaMin_ and etaMax_ are set correctly by user
	//or possible user only wants a limit in one directions
	//so should be able to set phi or eta to something extreme (-100,100) e.g.
	double dPhi = phi - vetoDir_.phi();
	double dEta = eta - vetoDir_.eta();
	while( dPhi < -M_PI ) 	dPhi += 2*M_PI;
	while( dPhi >= M_PI )   dPhi -= 2*M_PI;
    return (etaMin_ < dEta) && (dEta < etaMax_) && 
           (phiMin_ < dPhi) && (dPhi < phiMax_); 
}

void RectangularEtaPhiVeto::centerOn(double eta, double phi) { 
	vetoDir_ = Direction(eta,phi);
}
