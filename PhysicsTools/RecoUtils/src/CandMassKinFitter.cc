#include "PhysicsTools/RecoUtils/interface/CandMassKinFitter.h"
#include "PhysicsTools/KinFitter/interface/TKinFitter.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleEtEtaPhi.h"
#include "TMatrixD.h"
#include <iostream>
using namespace reco;
using namespace std;

FitQuality CandMassKinFitter::set(Candidate & c) const {
  TKinFitter fitter("CandMassFit", "CandMassFit");
  TString name("dau0");
  size_t daus = c.numberOfDaughters();
  vector<TMatrixD> errors(daus, TMatrix(3,3));
  vector<TLorentzVector> momenta(daus);
  vector<TFitParticleEtEtaPhi *> particles(daus, 0);
  TFitConstraintM constraint("MassConstraint", "MassConstraint", 0, 0 , mass_);
  for (size_t i = 0; i < daus; ++ i) {
    const Candidate & dau = * c.daughter(i);
    TMatrixD & err = errors[i];
    TLorentzVector & mom = momenta[i];
    mom.SetPtEtaPhiE(dau.pt(), dau.eta(), dau.phi(), dau.energy());
    err.Zero();
    err(0,0) = errEt(dau.et(), dau.eta());
    err(1,1) = errEta(dau.et(), dau.eta());
    err(2,2) = errPhi(dau.et(), dau.eta());
    fitter.addMeasParticle(particles[i] = new TFitParticleEtEtaPhi(name, name, & mom, & err));
    name[3]++;
    constraint.addParticle1(particles[i]);
  } 
  fitter.addConstraint(& constraint);
  fitter.setMaxNbIter(30);
  fitter.setMaxDeltaS(1e-2);
  fitter.setMaxF(1e-1);
  fitter.setVerbosity(0);			
  fitter.fit();
  // if (  fitter->getStatus() != 0 ) throw ...
  TLorentzVector sum(0, 0, 0, 0);
  for(size_t i = 0; i < daus; ++ i) {
    Candidate & dau = * c.daughter(i);
    TFitParticleEtEtaPhi * part =  particles[i];
    const TLorentzVector * p4 = part->getCurr4Vec();
    dau.setP4(Particle::LorentzVector(p4->X(), p4->Y(), p4->Z(), p4->T()));
    sum += * p4;
    delete particles[i];
  }
  c.setP4(Particle::LorentzVector(sum.X(), sum.Y(), sum.Z(), sum.T()));
  return FitQuality(fitter.getS(), fitter.getNDF());
}
