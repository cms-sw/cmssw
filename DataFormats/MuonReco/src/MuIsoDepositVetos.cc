#include "DataFormats/MuonReco/interface/MuIsoDepositVetos.h"
#include "DataFormats/MuonReco/interface/Direction.h"

#include <cmath>
#include <iostream>

using namespace std;
using namespace reco::isodeposit;

bool ConeVeto::veto(double eta, double phi, float value) const {
    return ( vetoDir_.deltaR2(eta,phi) < dR2_ );
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

bool ConeThresholdVeto::veto(double eta, double phi, float value) const {
    return (value <= threshold_) || ( vetoDir_.deltaR2(eta,phi) < dR2_ );
}
void ConeThresholdVeto::centerOn(double eta, double phi) { 
	vetoDir_ = Direction(eta,phi);
}

///////////////////////////////////////////////////////////////////////////

AngleConeVeto::AngleConeVeto(math::XYZVectorD dir, double angle) : vetoDir_(dir.Unit()), cosTheta_(cos(angle)) {
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

AngleCone::AngleCone(math::XYZVectorD dir, double angle) : coneDir_(dir.Unit()), cosTheta_(cos(angle)) {
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

bool RectangularEtaPhiVeto::veto(double eta, double phi, float value) const  {
    return (etaMin_ < eta) && (eta < etaMax_) && 
           (phiMin_ < phi) && (phi < phiMax_); 
}
