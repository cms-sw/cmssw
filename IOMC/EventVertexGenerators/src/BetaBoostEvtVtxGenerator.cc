
/*
  ________________________________________________________________________

  BetaBoostEvtVtxGenerator

  Smear vertex according to the Beta function on the transverse plane
  and a Gaussian on the z axis. It allows the beam to have a crossing
  angle (slopes dxdz and dydz).

  Based on GaussEvtVtxGenerator
  implemented by Francisco Yumiceva (yumiceva@fnal.gov)

  FERMILAB
  2006
  ________________________________________________________________________
*/

//lingshan: add beta for z-axis boost

#include "IOMC/EventVertexGenerators/interface/BetaBoostEvtVtxGenerator.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "HepMC/SimpleVector.h"
#include "TMatrixD.h"

#include <iostream>

using namespace edm;
using namespace std;
using namespace CLHEP;

BetaBoostEvtVtxGenerator::BetaBoostEvtVtxGenerator(edm::ParameterSet const& p, edm::ConsumesCollector& iC) :
  BaseEvtVtxGenerator(),
  fVertex(new HepMC::FourVector()), boost_(), fTimeOffset(0),
  verbosity_(p.getUntrackedParameter<bool>("verbosity",false)) {
  fX0 =        p.getParameter<double>("X0")*cm;
  fY0 =        p.getParameter<double>("Y0")*cm;
  fZ0 =        p.getParameter<double>("Z0")*cm;
  fSigmaZ =    p.getParameter<double>("SigmaZ")*cm;
  alpha_ =     p.getParameter<double>("Alpha")*radian;
  phi_ =       p.getParameter<double>("Phi")*radian;
  fbetastar =  p.getParameter<double>("BetaStar")*cm;
  femittance = p.getParameter<double>("Emittance")*cm; // this is not the normalized emittance
  fTimeOffset = p.getParameter<double>("TimeOffset")*ns*c_light; // HepMC time units are mm
  beta_=p.getParameter<double>("Beta"); 
  if (fSigmaZ <= 0) {
    throw cms::Exception("Configuration")
      << "Error in BetaBoostEvtVtxGenerator: "
      << "Illegal resolution in Z (SigmaZ is negative)";
  }
}

BetaBoostEvtVtxGenerator::~BetaBoostEvtVtxGenerator() {
}

void BetaBoostEvtVtxGenerator::generateNewVertex_(edm::HepMCProduct& product, CLHEP::HepRandomEngine& engine) {
   product.applyVtxGen(newVertex(engine));
   product.boostToLab(GetInvLorentzBoost(), "vertex");
   product.boostToLab(GetInvLorentzBoost(), "momentum");
}

HepMC::FourVector* BetaBoostEvtVtxGenerator::newVertex(CLHEP::HepRandomEngine& engine) {
  double tmp_sigz = CLHEP::RandGaussQ::shoot(&engine, 0.0, fSigmaZ);
  double Z = tmp_sigz + fZ0;

  double tmp_sigx = BetaFunction(Z,fZ0); 
  // need sqrt(2) for beamspot width relative to single beam width
  tmp_sigx /= sqrt(2.0);
  double X = CLHEP::RandGaussQ::shoot(&engine, 0.0, tmp_sigx) + fX0; // + Z*fdxdz;

  double tmp_sigy = BetaFunction(Z,fZ0);
  // need sqrt(2) for beamspot width relative to single beam width
  tmp_sigy /= sqrt(2.0);
  double Y = CLHEP::RandGaussQ::shoot(&engine, 0.0, tmp_sigy) + fY0; // + Z*fdydz;

  double tmp_sigt = CLHEP::RandGaussQ::shoot(&engine, 0.0, fSigmaZ);
  double T = tmp_sigt + fTimeOffset; 

  fVertex->set(X,Y,Z,T);
		
  return fVertex.get();
}

double BetaBoostEvtVtxGenerator::BetaFunction(double z, double z0) {
  return sqrt(femittance*(fbetastar+(((z-z0)*(z-z0))/fbetastar)));
}

void BetaBoostEvtVtxGenerator::sigmaZ(double s) { 
  if (s>=0) {
    fSigmaZ=s; 
  } else {
    throw cms::Exception("LogicError")
      << "Error in BetaBoostEvtVtxGenerator::sigmaZ: "
      << "Illegal resolution in Z (negative)";
  }
}

TMatrixD* BetaBoostEvtVtxGenerator::GetInvLorentzBoost() {

  //alpha_ = 0;
  //phi_ = 142.e-6;
  if (boost_) return boost_.get();
	
  //boost_.ResizeTo(4,4);
  //boost_ = new TMatrixD(4,4);

  TMatrixD tmpboost(4,4);
  TMatrixD tmpboostZ(4,4);
  TMatrixD tmpboostXYZ(4,4);

  //if ((alpha_ == 0) && (phi_==0)) { boost_->Zero(); return boost_; }
	
  // Lorentz boost to frame where the collision is head-on
  // phi is the half crossing angle in the plane ZS
  // alpha is the angle to the S axis from the X axis in the XY plane
	
  tmpboost(0,0) = 1./cos(phi_);
  tmpboost(0,1) = - cos(alpha_)*sin(phi_);
  tmpboost(0,2) = - tan(phi_)*sin(phi_);
  tmpboost(0,3) = - sin(alpha_)*sin(phi_);
  tmpboost(1,0) = - cos(alpha_)*tan(phi_);
  tmpboost(1,1) = 1.;
  tmpboost(1,2) = cos(alpha_)*tan(phi_);
  tmpboost(1,3) = 0.;
  tmpboost(2,0) = 0.;
  tmpboost(2,1) = - cos(alpha_)*sin(phi_);
  tmpboost(2,2) = cos(phi_);
  tmpboost(2,3) = - sin(alpha_)*sin(phi_);
  tmpboost(3,0) = - sin(alpha_)*tan(phi_);
  tmpboost(3,1) = 0.;
  tmpboost(3,2) = sin(alpha_)*tan(phi_);
  tmpboost(3,3) = 1.;
  //cout<<"beta "<<beta_;
  double gama=1.0/sqrt(1-beta_*beta_);
  tmpboostZ(0,0)=gama;
  tmpboostZ(0,1)=0.;
  tmpboostZ(0,2)=-1.0*beta_*gama;
  tmpboostZ(0,3)=0.;
  tmpboostZ(1,0)=0.;
  tmpboostZ(1,1) = 1.;
  tmpboostZ(1,2)=0.;
  tmpboostZ(1,3)=0.;
  tmpboostZ(2,0)=-1.0*beta_*gama;
  tmpboostZ(2,1) = 0.;
  tmpboostZ(2,2)=gama;
  tmpboostZ(2,3) = 0.;
  tmpboostZ(3,0)=0.;
  tmpboostZ(3,1)=0.;
  tmpboostZ(3,2)=0.;
  tmpboostZ(3,3) = 1.;

  tmpboostXYZ=tmpboostZ*tmpboost;
  tmpboostXYZ.Invert();

  boost_.reset(new TMatrixD(tmpboostXYZ));
  if (verbosity_) { boost_->Print(); }
	
  return boost_.get();
}

