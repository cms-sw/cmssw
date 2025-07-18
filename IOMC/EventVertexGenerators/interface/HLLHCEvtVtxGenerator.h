#ifndef IOMC_HLLHCEvtVtxGenerator_H
#define IOMC_HLLHCEvtVtxGenerator_H

/**
 * Generate event vertices given beams sizes, crossing angle
 * offset, and crab rotation. 
 * Attention: All values are assumed to be mm for spatial coordinates
 * and ns for time.
 * Attention: This class fix the the vertex time generation of HLLHCEvtVtxGenerator
 *
 * $Id: HLLHCEvtVtxGenerator_Fix.h,v 1.0 2015/03/15 10:34:38 Exp $
 */

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/DataRecord/interface/SimBeamSpotHLLHCObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotHLLHCObjects.h"

#include <string>

namespace CLHEP {
  class RandFlat;
}

namespace edm {
  class ConfigurationDescriptions;
}

class HLLHCEvtVtxGenerator : public BaseEvtVtxGenerator {
public:
  HLLHCEvtVtxGenerator(const edm::ParameterSet& p);

  /** Copy constructor */
  HLLHCEvtVtxGenerator(const HLLHCEvtVtxGenerator& p) = delete;

  /** Copy assignment operator */
  HLLHCEvtVtxGenerator& operator=(const HLLHCEvtVtxGenerator& rhs) = delete;

  ~HLLHCEvtVtxGenerator() override = default;

  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /// return a new event vertex
  ROOT::Math::XYZTVector vertexShift(CLHEP::HepRandomEngine*) const override;

  TMatrixD const* GetInvLorentzBoost() const override { return nullptr; };

private:
  // Configurable parameters
  double fMeanX, fMeanY, fMeanZ, fTimeOffset_c_light;  //spatial and time offset for mean collision
  double fEProton;                                     // proton beam energy
  double fCrossingAngle;                               // crossing angle
  double fCrabFrequency;                               // crab cavity frequency
  bool fRF800;                                         // 800 MHz RF?
  double fBetaCrossingPlane;                           // beta crossing plane (m)
  double fBetaSeparationPlane;                         // beta separation plane (m)
  double fHorizontalEmittance;                         // horizontal emittance
  double fVerticalEmittance;                           // vertical emittance
  double fBunchLength;                                 // bunch length
  double fCrabbingAngleCrossing;                       // crabbing angle crossing
  double fCrabbingAngleSeparation;                     // crabbing angle separation

  // Parameters inferred from configurables
  double gamma;  // beam configurations
  double beta;
  double betagamma;
  double oncc;   // ratio of crabbing angle to crossing angle
  double epsx;   // normalized crossing emittance
  double epss;   // normalized separation emittance
  double sigx;   // size in x
  double phiCR;  // crossing angle * crab frequency

  //width for y plane
  double sigma(double z, double epsilon, double beta, double betagamma) const;

  //density with crabbing
  double integrandCC(double x, double z, double t) const;

  // 4D intensity
  double intensity(double x, double y, double z, double t) const;

  // Read from DB
  bool readDB_;
  void update(const edm::EventSetup& iEventSetup);
  edm::ESWatcher<SimBeamSpotHLLHCObjectsRcd> parameterWatcher_;
  edm::ESGetToken<SimBeamSpotHLLHCObjects, SimBeamSpotHLLHCObjectsRcd> beamToken_;
};

#endif
