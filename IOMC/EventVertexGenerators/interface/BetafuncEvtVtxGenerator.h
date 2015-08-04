#ifndef IOMC_EventVertexGenerators_BetafuncEvtVtxGenerator_h
#define IOMC_EventVertexGenerators_BetafuncEvtVtxGenerator_h

/*
________________________________________________________________________

 BetafuncEvtVtxGenerator

 Smear vertex according to the Beta function on the transverse plane
 and a Gaussian on the z axis. It allows the beam to have a crossing
 angle (dx/dz and dy/dz).

 Based on GaussEvtVtxGenerator.h
 implemented by Francisco Yumiceva (yumiceva@fnal.gov)

 FERMILAB
 2006
________________________________________________________________________
*/

#include <memory>

#include "TMatrixD.h"

#include "GeneratorInterface/Core/interface/BaseEvtVtxGenerator.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "CondFormats/DataRecord/interface/SimBeamSpotObjectsRcd.h"

class BetafuncEvtVtxGenerator : public BaseEvtVtxGenerator {
public:
  BetafuncEvtVtxGenerator(edm::ParameterSet const& p, edm::ConsumesCollector& iC);
  virtual ~BetafuncEvtVtxGenerator();

  BetafuncEvtVtxGenerator(BetafuncEvtVtxGenerator const&) = delete;
  BetafuncEvtVtxGenerator& operator=(BetafuncEvtVtxGenerator const&) = delete;

private:
  virtual void beginRun_(edm::Run const&, edm::EventSetup const&) override;
  virtual void beginLuminosityBlock_(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  virtual void generateNewVertex_(edm::HepMCProduct& product, CLHEP::HepRandomEngine& engine) override;

  HepMC::FourVector* newVertex(CLHEP::HepRandomEngine&);
  TMatrixD* GetInvLorentzBoost();
    
  /// set resolution in Z in cm
  void sigmaZ(double s=1.0);

  /// set mean in X in cm
  void X0(double m=0) { fX0=m; }
  /// set mean in Y in cm
  void Y0(double m=0) { fY0=m; }
  /// set mean in Z in cm
  void Z0(double m=0) { fZ0=m; }

  /// set half crossing angle
  void Phi(double m=0) { phi_=m; }
  /// angle between crossing plane and horizontal plane
  void Alpha(double m=0) { alpha_=m; }

  /// set beta_star
  void betastar(double m=0) { fbetastar=m; }
  /// emittance (no the normalized)
  void emittance(double m=0) { femittance=m; }

  /// beta function
  double BetaFunction(double z, double z0);
  
  std::unique_ptr<HepMC::FourVector> fVertex;
  bool readDB_;

  double alpha_, phi_;
  std::unique_ptr<TMatrixD> boost_;
  
  double fX0, fY0, fZ0;
  double fSigmaZ;
  //double fdxdz, fdydz;
  double fbetastar, femittance;
  //  double falpha;
  double fTimeOffset;

  void update(edm::EventSetup const& iEventSetup);
  edm::ESWatcher<SimBeamSpotObjectsRcd> parameterWatcher_;
};

#endif
