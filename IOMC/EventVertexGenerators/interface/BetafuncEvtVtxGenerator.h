#ifndef IOMC_BetafuncEvtVtxGenerator_H
#define IOMC_BetafuncEvtVtxGenerator_H

// $Id: BetafuncEvtVtxGenerator.h,v 1.0 2006/07/20 14:34:40 yumiceva Exp $
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

namespace CLHEP {
   class RandGauss;
}

class BetafuncEvtVtxGenerator : public BaseEvtVtxGenerator 
{
public:
  BetafuncEvtVtxGenerator(const edm::ParameterSet & p);
  virtual ~BetafuncEvtVtxGenerator();

  /// return a new event vertex
  virtual CLHEP::Hep3Vector * newVertex();

  /// set resolution in Z in cm
  void sigmaZ(double s=1.0);

  /// set mean in X in cm
  void X0(double m=0) { fX0=m; }
  /// set mean in Y in cm
  void Y0(double m=0) { fY0=m; }
  /// set mean in Z in cm
  void Z0(double m=0) { fZ0=m; }

  /// set slope dxdz
  void dxdz(double m=0) { fdxdz=m; }
  /// set slope dydz
  void dydz(double m=0) { fdydz=m; }

  /// set beta_star
  void betastar(double m=0) { fbetastar=m; }
  void emmitance(double m=0) { femmitance=m; }

  /// beta function
  double BetaFunction(double z, double z0);
  
private:
  /** Copy constructor */
  BetafuncEvtVtxGenerator(const BetafuncEvtVtxGenerator &p);
  /** Copy assignment operator */
  BetafuncEvtVtxGenerator&  operator = (const BetafuncEvtVtxGenerator & rhs );
  
private:

  double fX0, fY0, fZ0;
  double fSigmaZ;
  double fdxdz, fdydz;
  double fbetastar, femmitance;
  
  CLHEP::RandGauss*  fRandom ;
  
};

#endif
