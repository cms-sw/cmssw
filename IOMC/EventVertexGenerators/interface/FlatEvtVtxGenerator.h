#ifndef IOMC_FlatEvtVtxGenerator_H
#define IOMC_FlatEvtVtxGenerator_H

/**
 * Generate event vertices according to a Flat distribution. 
 * Attention: All values are assumed to be cm!
 *
 * Important note: flat independent distributions in Z and T are not correct for physics production
 * In reality, if two flat beams interact the real distribution will not be flat with independent Z and T
 * but Z and T will be correlated, as example in GaussEvtVtxGenerator.
 * Can restore correlation in configuration via MinT += (MinZ - MaxZ)/2 and MaxT += (MaxZ - MinZ)/2
 * in [ns] units (recall c_light = 29.98cm/ns)
 *
 */

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

namespace CLHEP {
  class HepRandomEngine;
}

class FlatEvtVtxGenerator : public BaseEvtVtxGenerator {
public:
  FlatEvtVtxGenerator(const edm::ParameterSet& p);
  /** Copy constructor */
  FlatEvtVtxGenerator(const FlatEvtVtxGenerator& p) = delete;
  /** Copy assignment operator */
  FlatEvtVtxGenerator& operator=(const FlatEvtVtxGenerator& rhs) = delete;
  ~FlatEvtVtxGenerator() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /// return a new event vertex
  ROOT::Math::XYZTVector vertexShift(CLHEP::HepRandomEngine*) const override;

  const TMatrixD* GetInvLorentzBoost() const override { return nullptr; }

  /// set min in X in cm
  inline void minX(double m = 0.0) { fMinX = m; }
  /// set min in Y in cm
  inline void minY(double m = 0.0) { fMinY = m; }
  /// set min in Z in cm
  inline void minZ(double m = 0.0) { fMinZ = m; }

  /// set max in X in cm
  inline void maxX(double m = 0) { fMaxX = m; }
  /// set max in Y in cm
  inline void maxY(double m = 0) { fMaxY = m; }
  /// set max in Z in cm
  inline void maxZ(double m = 0) { fMaxZ = m; }

private:
  double fMinX, fMinY, fMinZ, fMinT, fMinR, fMinPhi;
  double fMaxX, fMaxY, fMaxZ, fMaxT, fMaxR, fMaxPhi;
  bool fFixedR;
};

#endif
