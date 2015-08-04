#ifndef IOMC_EventGenerators_FlatEvtVtxGenerator_h
#define IOMC_EventGenerators_FlatEvtVtxGenerator_h

/**
 * Generate event vertices according to a Flat distribution.
 * Attention: All values are assumed to be cm!
 *
 */

#include <memory>
#include "GeneratorInterface/Core/interface/BaseEvtVtxGenerator.h"

class FlatEvtVtxGenerator : public BaseEvtVtxGenerator {
public:
  FlatEvtVtxGenerator(edm::ParameterSet const& p, edm::ConsumesCollector& iC);
  virtual ~FlatEvtVtxGenerator();
  FlatEvtVtxGenerator(FlatEvtVtxGenerator const&) = delete;
  FlatEvtVtxGenerator& operator=(FlatEvtVtxGenerator const&) = delete;

private:
  virtual void generateNewVertex_(edm::HepMCProduct& product, CLHEP::HepRandomEngine& engine) override;

  HepMC::FourVector* newVertex(CLHEP::HepRandomEngine&);

  /// set min in X in cm
  void minX(double m=0.0);
  /// set min in Y in cm
  void minY(double m=0.0);
  /// set min in Z in cm
  void minZ(double m=0.0);

  /// set max in X in cm
  void maxX(double m=0);
  /// set max in Y in cm
  void maxY(double m=0);
  /// set max in Z in cm
  void maxZ(double m=0);

  std::unique_ptr<HepMC::FourVector> fVertex;
  double fMinX, fMinY, fMinZ;
  double fMaxX, fMaxY, fMaxZ;
  double fTimeOffset;
};

#endif
