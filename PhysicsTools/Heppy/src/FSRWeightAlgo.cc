#include "PhysicsTools/Heppy/interface/FSRWeightAlgo.h"


#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "CommonTools/CandUtils/interface/Booster.h"
#include <Math/VectorUtil.h>

namespace heppy {

double FSRWeightAlgo::weight() const {

  double weight = 1.;

  unsigned int gensize = genParticles_.size();
  for (unsigned int i = 0; i<gensize; ++i) {
    const reco::GenParticle& lepton = genParticles_[i];
    if (lepton.status()!=3) continue;
    int leptonId = lepton.pdgId();
    if (abs(leptonId)!=11 && abs(leptonId)!=13 && abs(leptonId)!=15) continue;
    if (lepton.numberOfMothers()!=1) continue;
    const reco::Candidate * boson = lepton.mother();
    int bosonId = abs(boson->pdgId());
    if (bosonId!=23  && bosonId!=24) continue;
    double bosonMass = boson->mass();
    double leptonMass = lepton.mass();
    double leptonEnergy = lepton.energy();
    double cosLeptonTheta = cos(lepton.theta());
    double sinLeptonTheta = sin(lepton.theta());
    double leptonPhi = lepton.phi();

    int trueKey = i;
    if (lepton.numberOfDaughters()==0) { 
      continue;
    } else if (lepton.numberOfDaughters()==1) { 
      int otherleptonKey = lepton.daughterRef(0).key();
      const reco::GenParticle& otherlepton = genParticles_[otherleptonKey];
      if (otherlepton.pdgId()!=leptonId) continue;
      if (otherlepton.numberOfDaughters()<=1) continue;
      trueKey = otherleptonKey;
    }

    const reco::GenParticle& trueLepton = genParticles_[trueKey];
    unsigned int nDaughters = trueLepton.numberOfDaughters();

    for (unsigned int j = 0; j<nDaughters; ++j) {
      const reco::Candidate * photon = trueLepton.daughter(j);
      if (photon->pdgId()!=22) continue;
      double photonEnergy = photon->energy();
      double cosPhotonTheta = cos(photon->theta());
      double sinPhotonTheta = sin(photon->theta());
      double photonPhi = photon->phi();
      double costheta = sinLeptonTheta*sinPhotonTheta*cos(leptonPhi-photonPhi)
	+ cosLeptonTheta*cosPhotonTheta;
      // Missing O(alpha) terms in soft-collinear approach
      // Only for W, from hep-ph/0303260
      if (bosonId==24) {
	double betaLepton = sqrt(1-pow(leptonMass/leptonEnergy,2));
	double delta = - 8*photonEnergy *(1-betaLepton*costheta)
	  / pow(bosonMass,3) 
	  / (1-pow(leptonMass/bosonMass,2))
	  / (4-pow(leptonMass/bosonMass,2))
	  * leptonEnergy * (pow(leptonMass,2)/bosonMass+2*photonEnergy);
	weight *= (1 + delta);
      }
      // Missing NLO QED orders in QED parton shower approach
      // Change coupling scale from 0 to kT to estimate this effect
      weight *= alphaRatio(photonEnergy*sqrt(1-pow(costheta,2)));
    }
  }

  return weight;
}


double FSRWeightAlgo::alphaRatio(double pt) const {

  double pigaga = 0.;

  // Leptonic contribution (just one loop, precise at < 0.3% level)
  const double alphapi = 1/137.036/M_PI;
  const double mass_e = 0.0005;
  const double mass_mu = 0.106;
  const double mass_tau = 1.777;
  const double mass_Z = 91.2;
  if (pt>mass_e) pigaga += alphapi * (2*log(pt/mass_e)/3.-5./9.);
  if (pt>mass_mu) pigaga += alphapi * (2*log(pt/mass_mu)/3.-5./9.);
  if (pt>mass_tau) pigaga += alphapi * (2*log(pt/mass_tau)/3.-5./9.);

  // Hadronic vaccum contribution
  // Using simple effective parametrization from Physics Letters B 513 (2001) 46.
  // Top contribution neglected
  double A = 0.; 
  double B = 0.; 
  double C = 0.; 
  if (pt<0.7) {
    A = 0.0; B = 0.0023092; C = 3.9925370;
  } else if (pt<2.0) {
    A = 0.0; B = 0.0022333; C = 4.2191779;
  } else if (pt<4.0) {
    A = 0.0; B = 0.0024402; C = 3.2496684;
  } else if (pt<10.0) {
    A = 0.0; B = 0.0027340; C = 2.0995092;
  } else if (pt<mass_Z) {
    A = 0.0010485; B = 0.0029431; C = 1.0;
  } else if (pt<10000.) {
    A = 0.0012234; B = 0.0029237; C = 1.0;
  } else {
    A = 0.0016894; B = 0.0028984; C = 1.0;
  }
  pigaga += A + B*log(1.+C*pt*pt);

  // Done
  return 1./(1.-pigaga);
}

}
