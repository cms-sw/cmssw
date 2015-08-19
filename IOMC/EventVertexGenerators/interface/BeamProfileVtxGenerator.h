#ifndef IOMC_EventVertexGenerator_BeamProfileVtxGenerator_h
#define IOMC_EventVertexGenerator_BeamProfileVtxGenerator_h

/**
 * Generate event vertices according to a Gaussian distribution transverse
 * to beam direction (given by eta and phi
 * Attention: Units are assumed to be cm and radian!
 * \author Sunanda Banerjee
 *
 */

#include "GeneratorInterface/Core/interface/BaseEvtVtxGenerator.h"
#include <memory>
#include <vector>

class BeamProfileVtxGenerator : public BaseEvtVtxGenerator {
public:
  BeamProfileVtxGenerator(edm::ParameterSet const& p, edm::ConsumesCollector& iC);
  virtual ~BeamProfileVtxGenerator();
  BeamProfileVtxGenerator(BeamProfileVtxGenerator const&) = delete;
  BeamProfileVtxGenerator& operator=(BeamProfileVtxGenerator const& rhs) = delete;

  /// set resolution in X in cm
  void sigmaX(double s=1.0);
  /// set resolution in Y in cm
  void sigmaY(double s=1.0);

  /// set mean in X in cm
  void meanX(double m=0)   {fMeanX=m;}
  /// set mean in Y in cm
  void meanY(double m=0)   {fMeanY=m;}
  /// set mean in Z in cm
  void beamPos(double m=0) {fMeanZ=m;}

  /// set eta
  void eta(double m=0);
  /// set phi in radian
  void phi(double m=0)     {fPhi=m;}
  /// set psi in radian
  void psi(double m=999)     {fPsi=m;}
  /// set type
  void setType(bool m=true);
  
private:
  virtual void generateNewVertex_(edm::HepMCProduct& product, CLHEP::HepRandomEngine& engine) override;

  /// return a new event vertex
  HepMC::FourVector* newVertex(CLHEP::HepRandomEngine&);

  std::unique_ptr<HepMC::FourVector> fVertex;
  double      fSigmaX, fSigmaY;
  double      fMeanX,  fMeanY, fMeanZ;
  double      fEta,    fPhi,   fTheta;

  double      fPsi;

  bool        fType,   ffile;
  int         nBinx,   nBiny;
  std::vector<double> fdistn;
  double fTimeOffset;
};

#endif
