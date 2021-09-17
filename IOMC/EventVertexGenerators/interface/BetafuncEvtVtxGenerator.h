#ifndef IOMC_BetafuncEvtVtxGenerator_H
#define IOMC_BetafuncEvtVtxGenerator_H

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

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/DataRecord/interface/SimBeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"

namespace CLHEP {
  class HepRandomEngine;
}

class BetafuncEvtVtxGenerator : public BaseEvtVtxGenerator {
public:
  BetafuncEvtVtxGenerator(const edm::ParameterSet& p);
  /** Copy constructor */
  BetafuncEvtVtxGenerator(const BetafuncEvtVtxGenerator& p) = delete;
  /** Copy assignment operator */
  BetafuncEvtVtxGenerator& operator=(const BetafuncEvtVtxGenerator& rhs) = delete;
  ~BetafuncEvtVtxGenerator() override;

  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  /// return a new event vertex
  //virtual CLHEP::Hep3Vector * newVertex();
  HepMC::FourVector newVertex(CLHEP::HepRandomEngine*) const override;

  TMatrixD const* GetInvLorentzBoost() const override;

  /// set resolution in Z in cm
  void sigmaZ(double s = 1.0);

  /// set mean in X in cm
  void X0(double m = 0) { fX0 = m; }
  /// set mean in Y in cm
  void Y0(double m = 0) { fY0 = m; }
  /// set mean in Z in cm
  void Z0(double m = 0) { fZ0 = m; }

  /// set beta_star
  void betastar(double m = 0) { fbetastar = m; }
  /// emittance (no the normalized)
  void emittance(double m = 0) { femittance = m; }

  /// beta function
  double BetaFunction(double z, double z0) const;

private:
  void setBoost(double alpha, double phi);

private:
  bool readDB_;

  double fX0, fY0, fZ0;
  double fSigmaZ;
  //double fdxdz, fdydz;
  double fbetastar, femittance;
  //  double falpha;
  double fTimeOffset;

  TMatrixD boost_;

  void update(const edm::EventSetup& iEventSetup);
  edm::ESWatcher<SimBeamSpotObjectsRcd> parameterWatcher_;
  edm::ESGetToken<SimBeamSpotObjects, SimBeamSpotObjectsRcd> beamToken_;
};

#endif
