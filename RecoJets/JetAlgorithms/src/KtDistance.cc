#include "RecoJets/JetAlgorithms/interface/KtDistance.h"
#include "RecoJets/JetAlgorithms/interface/KtUtil.h"
#include "RecoJets/JetAlgorithms/interface/KtDistanceInterface.h"
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
namespace KtJet {


KtDistance* getDistanceScheme(int angle, int collision_type) {
  if (angle == 1)      return new KtDistanceAngle(collision_type);
  else if( angle == 2) return new KtDistanceDeltaR(collision_type);
  else if (angle == 3) return new KtDistanceQCD(collision_type);
  else{
    std::cout << "[Jets] WARNING, unreconised distance scheme specified!" << std::endl;
    std::cout << "[Jets] Distance Scheme set to KtDistanceAngle" << std::endl;
    return new KtDistanceAngle(collision_type);
  }
}

KtDistanceAngle::KtDistanceAngle(int collision_type) : m_type(collision_type), m_name("angle") {}
  //KtDistanceAngle::~KtDistanceAngle() {}
std::string KtDistanceAngle::name() const {return m_name;}

KtFloat KtDistanceAngle::operator()(const KtLorentzVector & a) const {
  KtFloat kt, r, costh;
  const KtFloat small = 0.0001;     // ??? Should be defined somewhere else?
  switch (m_type) {            // direction of beam depends on collision type
  case 1:
    return -1;               // e+e- : no beam remnant, so result will be ignored anyway
    break;
  case 2:                    // ep (p beam -z direction)
    costh = -(a.cosTheta());
    break;
  case 3:                    // pe (p beam +z direction)
    costh = a.cosTheta();
    break;
  case 4:                    // pp (p beams in both directions)
    costh = fabs(a.cosTheta());
    break;
  default:                   // type out of range - WARNING ???
    costh = 0.;
    break;
  }
  r = 2*(1-costh);
  if (r<small) r = a.perp2()/a.vect().mag2();  // Use approx if close to beam
  kt = a.e()*a.e() * r;
  return kt;
}

KtFloat KtDistanceAngle::operator()(const KtLorentzVector & a, const KtLorentzVector & b) const {
  KtFloat emin = std::min(a.e(),b.e());
  KtFloat esq = emin*emin;
  KtFloat costh = a.vect().cosTheta(b.vect());
  return 2 * esq * (1 - costh);
}


KtDistanceDeltaR::KtDistanceDeltaR(int collision_type) : m_type(collision_type), m_name("DeltaR") {}
  //KtDistanceDeltaR::~KtDistanceDeltaR() {}
std::string KtDistanceDeltaR::name() const {return m_name;}

KtFloat KtDistanceDeltaR::operator()(const KtLorentzVector & a) const {
  return (m_type==1) ? -1 : a.perp2(); // If e+e-, no beam remnant, so result will be ignored anyway
}

KtFloat KtDistanceDeltaR::operator()(const KtLorentzVector & a, const KtLorentzVector & b) const {
  KtFloat rsq,esq,kt,deltaEta,deltaPhi;
  deltaEta = a.crapidity()-b.crapidity();
  deltaPhi = phiAngle(a.phi()-b.phi());
  rsq = deltaEta*deltaEta + deltaPhi*deltaPhi;
  esq = std::min(a.perp2(),b.perp2());
  kt = esq*rsq;
  return kt;
}


KtDistanceQCD::KtDistanceQCD(int collision_type) : m_type(collision_type), m_name("QCD") {}
  //KtDistanceQCD::~KtDistanceQCD() {}
std::string KtDistanceQCD::name() const {return m_name;}

KtFloat KtDistanceQCD::operator()(const KtLorentzVector & a) const {
  return (m_type==1) ? -1 : a.perp2(); // If e+e-, no beam remnant, so result will be ignored anyway
}

KtFloat KtDistanceQCD::operator()(const KtLorentzVector & a, const KtLorentzVector & b) const {
  KtFloat rsq,esq,kt,deltaEta,deltaPhi;
  deltaEta = a.crapidity()-b.crapidity();
  deltaPhi = phiAngle(a.phi()-b.phi());
  rsq = 2 * (cosh(deltaEta)-cos(deltaPhi));
  esq = std::min(a.perp2(),b.perp2());
  kt = esq*rsq;
  return kt;
}

}//end of namespace
