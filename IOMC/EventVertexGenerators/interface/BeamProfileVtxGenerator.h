#ifndef IOMC_BeamProfileVtxGenerator_H
#define IOMC_BeamProfileVtxGenerator_H

/**
 * Generate event vertices according to a Gaussian distribution transverse
 * to beam direction (given by eta and phi
 * Attention: Units are assumed to be cm and radian!
 * \author Sunanda Banerjee
 *
 * $Id: BeamProfileVtxGenerator.h,v 1.8 2009/08/29 00:12:38 sunanda Exp $
 */

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"
#include <vector>

namespace CLHEP
{
   class HepRandom;
}

class BeamProfileVtxGenerator : public BaseEvtVtxGenerator
{
public:
  BeamProfileVtxGenerator(const edm::ParameterSet & p);
  virtual ~BeamProfileVtxGenerator();

  /// return a new event vertex
  //virtual CLHEP::Hep3Vector * newVertex();
  virtual HepMC::FourVector* newVertex() ;

  virtual TMatrixD* GetInvLorentzBoost() {
	  return 0;
  }

    
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
  /** Copy constructor */
  BeamProfileVtxGenerator(const BeamProfileVtxGenerator &p);
  /** Copy assignment operator */
  BeamProfileVtxGenerator& operator = (const BeamProfileVtxGenerator& rhs);
private:
  double      fSigmaX, fSigmaY;
  double      fMeanX,  fMeanY, fMeanZ;
  double      fEta,    fPhi,   fTheta;

  double      fPsi;

  bool        fType,   ffile;
  int         nBinx,   nBiny;
  std::vector<double> fdistn;
  CLHEP::HepRandom*   fRandom;
  double fTimeOffset;
};

#endif
