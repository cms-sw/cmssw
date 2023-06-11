
/*
________________________________________________________________________

 BetafuncEvtVtxGenerator

 Smear vertex according to the Beta function on the transverse plane
 and a Gaussian on the z axis. It allows the beam to have a crossing
 angle (slopes dxdz and dydz).

 Based on GaussEvtVtxGenerator
 implemented by Francisco Yumiceva (yumiceva@fnal.gov)

 FERMILAB
 2006
________________________________________________________________________
*/

#include "IOMC/EventVertexGenerators/interface/BetafuncEvtVtxGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
//#include "CLHEP/Vector/ThreeVector.h"
#include "HepMC/SimpleVector.h"

#include <iostream>

BetafuncEvtVtxGenerator::BetafuncEvtVtxGenerator(const edm::ParameterSet& p) : BaseEvtVtxGenerator(p), boost_(4, 4) {
  readDB_ = p.getParameter<bool>("readDB");
  if (!readDB_) {
    fX0 = p.getParameter<double>("X0") * cm;
    fY0 = p.getParameter<double>("Y0") * cm;
    fZ0 = p.getParameter<double>("Z0") * cm;
    fSigmaZ = p.getParameter<double>("SigmaZ") * cm;
    fbetastar = p.getParameter<double>("BetaStar") * cm;
    femittance = p.getParameter<double>("Emittance") * cm;              // this is not the normalized emittance
    fTimeOffset = p.getParameter<double>("TimeOffset") * ns * c_light;  // HepMC time units are mm

    setBoost(p.getParameter<double>("Alpha") * radian, p.getParameter<double>("Phi") * radian);
    if (fSigmaZ <= 0) {
      throw cms::Exception("Configuration") << "Error in BetafuncEvtVtxGenerator: "
                                            << "Illegal resolution in Z (SigmaZ is negative)";
    }
  }
  if (readDB_) {
    beamToken_ = esConsumes<SimBeamSpotObjects, SimBeamSpotObjectsRcd, edm::Transition::BeginLuminosityBlock>();
  }
}

BetafuncEvtVtxGenerator::~BetafuncEvtVtxGenerator() {}

void BetafuncEvtVtxGenerator::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const& iEventSetup) {
  update(iEventSetup);
}

void BetafuncEvtVtxGenerator::update(const edm::EventSetup& iEventSetup) {
  if (readDB_ && parameterWatcher_.check(iEventSetup)) {
    edm::ESHandle<SimBeamSpotObjects> beamhandle = iEventSetup.getHandle(beamToken_);

    fX0 = beamhandle->x();
    fY0 = beamhandle->y();
    fZ0 = beamhandle->z();
    //    falpha=beamhandle->alpha();
    fSigmaZ = beamhandle->sigmaZ();
    fTimeOffset = beamhandle->timeOffset();
    fbetastar = beamhandle->betaStar();
    femittance = beamhandle->emittance();
    setBoost(beamhandle->alpha(), beamhandle->phi());
  }
}

HepMC::FourVector BetafuncEvtVtxGenerator::newVertex(CLHEP::HepRandomEngine* engine) const {
  double X, Y, Z;

  double tmp_sigz = CLHEP::RandGaussQ::shoot(engine, 0., fSigmaZ);
  Z = tmp_sigz + fZ0;

  double tmp_sigx = BetaFunction(Z, fZ0);
  // need sqrt(2) for beamspot width relative to single beam width
  tmp_sigx /= sqrt(2.0);
  X = CLHEP::RandGaussQ::shoot(engine, 0., tmp_sigx) + fX0;  // + Z*fdxdz ;

  double tmp_sigy = BetaFunction(Z, fZ0);
  // need sqrt(2) for beamspot width relative to single beam width
  tmp_sigy /= sqrt(2.0);
  Y = CLHEP::RandGaussQ::shoot(engine, 0., tmp_sigy) + fY0;  // + Z*fdydz;

  double tmp_sigt = CLHEP::RandGaussQ::shoot(engine, 0., fSigmaZ);
  double T = tmp_sigt + fTimeOffset;

  return HepMC::FourVector(X, Y, Z, T);
}

double BetafuncEvtVtxGenerator::BetaFunction(double z, double z0) const {
  return sqrt(femittance * (fbetastar + (((z - z0) * (z - z0)) / fbetastar)));
}

void BetafuncEvtVtxGenerator::setBoost(double alpha, double phi) {
  //boost_.ResizeTo(4,4);
  //boost_ = new TMatrixD(4,4);
  TMatrixD tmpboost(4, 4);

  //if ( (alpha_ == 0) && (phi_==0) ) { boost_->Zero(); return boost_; }

  // Lorentz boost to frame where the collision is head-on
  // phi is the half crossing angle in the plane ZS
  // alpha is the angle to the S axis from the X axis in the XY plane

  tmpboost(0, 0) = 1. / cos(phi);
  tmpboost(0, 1) = -cos(alpha) * sin(phi);
  tmpboost(0, 2) = -tan(phi) * sin(phi);
  tmpboost(0, 3) = -sin(alpha) * sin(phi);
  tmpboost(1, 0) = -cos(alpha) * tan(phi);
  tmpboost(1, 1) = 1.;
  tmpboost(1, 2) = cos(alpha) * tan(phi);
  tmpboost(1, 3) = 0.;
  tmpboost(2, 0) = 0.;
  tmpboost(2, 1) = -cos(alpha) * sin(phi);
  tmpboost(2, 2) = cos(phi);
  tmpboost(2, 3) = -sin(alpha) * sin(phi);
  tmpboost(3, 0) = -sin(alpha) * tan(phi);
  tmpboost(3, 1) = 0.;
  tmpboost(3, 2) = sin(alpha) * tan(phi);
  tmpboost(3, 3) = 1.;

  tmpboost.Invert();
  boost_ = tmpboost;
  //boost_->Print();
}

void BetafuncEvtVtxGenerator::sigmaZ(double s) {
  if (s >= 0) {
    fSigmaZ = s;
  } else {
    throw cms::Exception("LogicError") << "Error in BetafuncEvtVtxGenerator::sigmaZ: "
                                       << "Illegal resolution in Z (negative)";
  }
}

TMatrixD const* BetafuncEvtVtxGenerator::GetInvLorentzBoost() const { return &boost_; }
