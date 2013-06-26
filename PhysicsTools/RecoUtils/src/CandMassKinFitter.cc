#include "PhysicsTools/RecoUtils/interface/CandMassKinFitter.h"
#include "PhysicsTools/KinFitter/interface/TKinFitter.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "PhysicsTools/KinFitter/interface/TFitConstraintM.h"
#include "PhysicsTools/KinFitter/interface/TFitParticleMCCart.h"
#include "TMatrixD.h"
#include <iostream>
using namespace reco;
using namespace std;

FitQuality CandMassKinFitter::set( Candidate & c ) const {
  TKinFitter fitter("CandMassFit", "CandMassFit");
  TString name("dau0");
  size_t daus = c.numberOfDaughters();
  vector<TMatrixD> errors(daus, TMatrix(3,3));
  vector<TVector3> momenta(daus);
  vector<TFitParticleMCCart *> particles(daus, 0);
  TFitConstraintM constraint( "MassConstraint", 
			      "MassConstraint", 0, 0 , mass_);
  for ( size_t i = 0; i < daus; ++ i ) {
    const Candidate & dau = * c.daughter( i );
    Particle::LorentzVector p4 = dau.p4();
    TMatrixD & err = errors[i];
    TVector3 & mom = momenta[i];
    mom = TVector3( p4.px(), p4.py(), p4.pz() );
    TrackRef trk = dau.get<TrackRef>();
    // dummy errors for now...
    // should play with track parametrization...
    err.Zero();
    err(0,0) = 0.1;
    err(1,1) = 0.1;
    err(2,2) = 0.1;
    fitter.addMeasParticle( particles[i] = new TFitParticleMCCart( name, name, & mom, dau.mass(), & err ) );
    name[3] ++;
    constraint.addParticle1( particles[i] );
  } 
  fitter.addConstraint(& constraint);
  fitter.setMaxNbIter( 30 );
  fitter.setMaxDeltaS( 1e-2 );
  fitter.setMaxF( 1e-1 );
  fitter.setVerbosity( 0 );			
  fitter.fit();
  // if (  fitter->getStatus() != 0 ) throw ...
  TLorentzVector sum( 0, 0, 0, 0 );
  for( size_t i = 0; i < daus; ++ i ) {
    Candidate & dau = * c.daughter( i );
    TFitParticleMCCart * part =  particles[i];
    const TLorentzVector * p4 = part->getCurr4Vec();
    dau.setP4( Particle::LorentzVector( p4->X(), p4->Y(), p4->Z(), p4->T() ) );
    sum += * p4;
    delete particles[i];
  }
  c.setP4( Particle::LorentzVector( sum.X(), sum.Y(), sum.Z(), sum.T() ) );
  return FitQuality( fitter.getS(), fitter.getNDF());
}
