#ifndef IOMC_EventVertexGenerators_GaussEvtVtxGenerator_h
#define IOMC_EventVertexGenerators_GaussEvtVtxGenerator_h

/**
 * Generate event vertices according to a Gauss distribution. 
 * Attention: All values are assumed to be cm!
 *
 */

#include <memory>

#include "GeneratorInterface/Core/interface/BaseEvtVtxGenerator.h"

class GaussEvtVtxGenerator : public BaseEvtVtxGenerator {
public:
  GaussEvtVtxGenerator(edm::ParameterSet const& p, edm::ConsumesCollector& iC);
  virtual ~GaussEvtVtxGenerator();

  GaussEvtVtxGenerator(GaussEvtVtxGenerator const&) = delete;
  GaussEvtVtxGenerator& operator = (GaussEvtVtxGenerator const&) = delete;

private:
  virtual void generateNewVertex_(edm::HepMCProduct& product, CLHEP::HepRandomEngine& engine) override;

  HepMC::FourVector* newVertex(CLHEP::HepRandomEngine&) ;

  /// set resolution in X in cm
  void sigmaX(double s=1.0);
  /// set resolution in Y in cm
  void sigmaY(double s=1.0);
  /// set resolution in Z in cm
  void sigmaZ(double s=1.0);

  /// set mean in X in cm
  void meanX(double m=0) { fMeanX=m; }
  /// set mean in Y in cm
  void meanY(double m=0) { fMeanY=m; }
  /// set mean in Z in cm
  void meanZ(double m=0) { fMeanZ=m; }
  
  std::unique_ptr<HepMC::FourVector> fVertex;
  double fSigmaX, fSigmaY, fSigmaZ;
  double fMeanX,  fMeanY,  fMeanZ;
  double fTimeOffset;
};

#endif
