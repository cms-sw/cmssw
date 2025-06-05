
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

//#include "IOMC/EventVertexGenerators/interface/BetafuncEvtVtxGenerator.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/AbstractServices/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <CLHEP/Random/RandGaussQ.h>
#include <CLHEP/Units/SystemOfUnits.h>
#include <CLHEP/Units/GlobalPhysicalConstants.h>
//#include "CLHEP/Vector/ThreeVector.h"
#include "HepMC/SimpleVector.h"
#include "TMatrixD.h"

#include <iostream>

using namespace edm;
using namespace std;
using namespace CLHEP;

namespace CLHEP {
  class HepRandomEngine;
}

class BetaBoostEvtVtxGenerator : public edm::global::EDProducer<> {
public:
  BetaBoostEvtVtxGenerator(const edm::ParameterSet& p);
  /** Copy constructor */
  BetaBoostEvtVtxGenerator(const BetaBoostEvtVtxGenerator& p) = delete;
  /** Copy assignment operator */
  BetaBoostEvtVtxGenerator& operator=(const BetaBoostEvtVtxGenerator& rhs) = delete;
  ~BetaBoostEvtVtxGenerator() override = default;

  /// return a new event vertex
  HepMC::FourVector newVertex(CLHEP::HepRandomEngine*) const;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  TMatrixD GetInvLorentzBoost() const;

  /// beta function
  double BetaFunction(double z, double z0) const;

private:
  const double alpha_;  // angle between crossing plane and horizontal plane
  const double phi_;    // half crossing angle
  const double beta_;
  const double fX0;      // mean in X in cm
  const double fY0;      // mean in Y in cm
  const double fZ0;      // mean in Z in cm
  const double fSigmaZ;  // resolution in Z in cm
  //double fdxdz, fdydz;
  const double fbetastar;
  const double femittance;  // emittance (no the normalized)a

  const double fTimeOffset;

  const TMatrixD boost_;

  const edm::EDGetTokenT<HepMCProduct> sourceLabel;

  const bool verbosity_;
};

BetaBoostEvtVtxGenerator::BetaBoostEvtVtxGenerator(const edm::ParameterSet& p)
    : alpha_(p.getParameter<double>("Alpha") * radian),
      phi_(p.getParameter<double>("Phi") * radian),
      beta_(p.getParameter<double>("Beta")),
      fX0(p.getParameter<double>("X0") * cm),
      fY0(p.getParameter<double>("Y0") * cm),
      fZ0(p.getParameter<double>("Z0") * cm),
      fSigmaZ(p.getParameter<double>("SigmaZ") * cm),
      fbetastar(p.getParameter<double>("BetaStar") * cm),
      femittance(p.getParameter<double>("Emittance") * cm),              // this is not the normalized emittance
      fTimeOffset(p.getParameter<double>("TimeOffset") * ns * c_light),  // HepMC time units are mm
      boost_(GetInvLorentzBoost()),
      sourceLabel(consumes<HepMCProduct>(p.getParameter<edm::InputTag>("src"))),
      verbosity_(p.getUntrackedParameter<bool>("verbosity", false)) {
  if (fSigmaZ <= 0) {
    throw cms::Exception("Configuration") << "Error in BetaBoostEvtVtxGenerator: "
                                          << "Illegal resolution in Z (SigmaZ is negative)";
  }

  produces<edm::HepMCProduct>();
}

HepMC::FourVector BetaBoostEvtVtxGenerator::newVertex(CLHEP::HepRandomEngine* engine) const {
  double X, Y, Z;

  double tmp_sigz = CLHEP::RandGaussQ::shoot(engine, 0.0, fSigmaZ);
  Z = tmp_sigz + fZ0;

  double tmp_sigx = BetaFunction(Z, fZ0);
  // need sqrt(2) for beamspot width relative to single beam width
  tmp_sigx /= sqrt(2.0);
  X = CLHEP::RandGaussQ::shoot(engine, 0.0, tmp_sigx) + fX0;  // + Z*fdxdz ;

  double tmp_sigy = BetaFunction(Z, fZ0);
  // need sqrt(2) for beamspot width relative to single beam width
  tmp_sigy /= sqrt(2.0);
  Y = CLHEP::RandGaussQ::shoot(engine, 0.0, tmp_sigy) + fY0;  // + Z*fdydz;

  double tmp_sigt = CLHEP::RandGaussQ::shoot(engine, 0.0, fSigmaZ);
  double T = tmp_sigt + fTimeOffset;

  return HepMC::FourVector(X, Y, Z, T);
}

double BetaBoostEvtVtxGenerator::BetaFunction(double z, double z0) const {
  return sqrt(femittance * (fbetastar + (((z - z0) * (z - z0)) / fbetastar)));
}

TMatrixD BetaBoostEvtVtxGenerator::GetInvLorentzBoost() const {
  //alpha_ = 0;
  //phi_ = 142.e-6;
  //	if (boost_ != 0 ) return boost_;

  //boost_.ResizeTo(4,4);
  //boost_ = new TMatrixD(4,4);
  TMatrixD tmpboost(4, 4);
  TMatrixD tmpboostZ(4, 4);
  TMatrixD tmpboostXYZ(4, 4);

  //if ( (alpha_ == 0) && (phi_==0) ) { boost_->Zero(); return boost_; }

  // Lorentz boost to frame where the collision is head-on
  // phi is the half crossing angle in the plane ZS
  // alpha is the angle to the S axis from the X axis in the XY plane

  tmpboost(0, 0) = 1. / cos(phi_);
  tmpboost(0, 1) = -cos(alpha_) * sin(phi_);
  tmpboost(0, 2) = -tan(phi_) * sin(phi_);
  tmpboost(0, 3) = -sin(alpha_) * sin(phi_);
  tmpboost(1, 0) = -cos(alpha_) * tan(phi_);
  tmpboost(1, 1) = 1.;
  tmpboost(1, 2) = cos(alpha_) * tan(phi_);
  tmpboost(1, 3) = 0.;
  tmpboost(2, 0) = 0.;
  tmpboost(2, 1) = -cos(alpha_) * sin(phi_);
  tmpboost(2, 2) = cos(phi_);
  tmpboost(2, 3) = -sin(alpha_) * sin(phi_);
  tmpboost(3, 0) = -sin(alpha_) * tan(phi_);
  tmpboost(3, 1) = 0.;
  tmpboost(3, 2) = sin(alpha_) * tan(phi_);
  tmpboost(3, 3) = 1.;
  //cout<<"beta "<<beta_;
  double gama = 1.0 / sqrt(1 - beta_ * beta_);
  tmpboostZ(0, 0) = gama;
  tmpboostZ(0, 1) = 0.;
  tmpboostZ(0, 2) = -1.0 * beta_ * gama;
  tmpboostZ(0, 3) = 0.;
  tmpboostZ(1, 0) = 0.;
  tmpboostZ(1, 1) = 1.;
  tmpboostZ(1, 2) = 0.;
  tmpboostZ(1, 3) = 0.;
  tmpboostZ(2, 0) = -1.0 * beta_ * gama;
  tmpboostZ(2, 1) = 0.;
  tmpboostZ(2, 2) = gama;
  tmpboostZ(2, 3) = 0.;
  tmpboostZ(3, 0) = 0.;
  tmpboostZ(3, 1) = 0.;
  tmpboostZ(3, 2) = 0.;
  tmpboostZ(3, 3) = 1.;

  tmpboostXYZ = tmpboostZ * tmpboost;
  tmpboostXYZ.Invert();

  if (verbosity_) {
    tmpboostXYZ.Print();
  }

  return tmpboostXYZ;
}

void BetaBoostEvtVtxGenerator::produce(edm::StreamID, Event& evt, const EventSetup&) const {
  edm::Service<RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration")
        << "Attempt to get a random engine when the RandomNumberGeneratorService is not configured.\n"
           "You must configure the service if you want an engine.\n";
  }
  CLHEP::HepRandomEngine* engine = &rng->getEngine(evt.streamID());

  const auto& HepUnsmearedMCEvt = evt.get(sourceLabel);

  // Copy the HepMC::GenEvent
  auto HepMCEvt = std::make_unique<edm::HepMCProduct>(new HepMC::GenEvent(*HepUnsmearedMCEvt.GetEvent()));

  // generate new vertex & apply the shift
  //
  auto vertex = newVertex(engine);
  HepMCEvt->applyVtxGen(&vertex);

  //HepMCEvt->LorentzBoost( 0., 142.e-6 );
  HepMCEvt->boostToLab(&boost_, "vertex");
  HepMCEvt->boostToLab(&boost_, "momentum");
  evt.put(std::move(HepMCEvt));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BetaBoostEvtVtxGenerator);
