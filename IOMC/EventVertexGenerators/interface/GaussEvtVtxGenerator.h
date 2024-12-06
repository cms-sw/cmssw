#ifndef IOMC_GaussEvtVtxGenerator_H
#define IOMC_GaussEvtVtxGenerator_H

/**
 * Generate event vertices according to a Gauss distribution. 
 * Attention: All values are assumed to be cm!
 *
 */

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/DataRecord/interface/SimBeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"

namespace CLHEP {
  class HepRandomEngine;
}

class GaussEvtVtxGenerator : public BaseEvtVtxGenerator {
public:
  GaussEvtVtxGenerator(const edm::ParameterSet& p);
  /** Copy constructor */
  GaussEvtVtxGenerator(const GaussEvtVtxGenerator& p) = delete;
  /** Copy assignment operator */
  GaussEvtVtxGenerator& operator=(const GaussEvtVtxGenerator& rhs) = delete;
  ~GaussEvtVtxGenerator() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  /// return a new event vertex
  ROOT::Math::XYZTVector vertexShift(CLHEP::HepRandomEngine*) const override;

  TMatrixD const* GetInvLorentzBoost() const override { return nullptr; }

  /// set resolution in X in cm
  void sigmaX(double s = 1.0);
  /// set resolution in Y in cm
  void sigmaY(double s = 1.0);
  /// set resolution in Z in cm
  void sigmaZ(double s = 1.0);

  /// set mean in X in cm
  void meanX(double m = 0) { fMeanX = m; }
  /// set mean in Y in cm
  void meanY(double m = 0) { fMeanY = m; }
  /// set mean in Z in cm
  void meanZ(double m = 0) { fMeanZ = m; }

private:
  bool readDB_;

  double fSigmaX, fSigmaY, fSigmaZ;
  double fMeanX, fMeanY, fMeanZ;
  double fTimeOffset;

  void update(const edm::EventSetup& iEventSetup);
  edm::ESWatcher<SimBeamSpotObjectsRcd> parameterWatcher_;
  edm::ESGetToken<SimBeamSpotObjects, SimBeamSpotObjectsRcd> beamToken_;
};

#endif
