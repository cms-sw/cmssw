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

  /// return a new event vertex
  //virtual CLHEP::Hep3Vector* newVertex();
  HepMC::FourVector newVertex(CLHEP::HepRandomEngine*) const override;

  const TMatrixD* GetInvLorentzBoost() const override { return nullptr; }

  /// set min in X in cm
  void minX(double m = 0.0);
  /// set min in Y in cm
  void minY(double m = 0.0);
  /// set min in Z in cm
  void minZ(double m = 0.0);

  /// set max in X in cm
  void maxX(double m = 0);
  /// set max in Y in cm
  void maxY(double m = 0);
  /// set max in Z in cm
  void maxZ(double m = 0);

private:
  double fMinX, fMinY, fMinZ, fMinT;
  double fMaxX, fMaxY, fMaxZ, fMaxT;
};

#endif
