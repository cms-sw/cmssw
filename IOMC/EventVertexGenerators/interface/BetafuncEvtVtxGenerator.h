#ifndef IOMC_BetafuncEvtVtxGenerator_H
#define IOMC_BetafuncEvtVtxGenerator_H

// $Id: BetafuncEvtVtxGenerator.h,v 1.5 2007/09/14 08:31:56 fabiocos Exp $
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
   class RandGaussQ;
}

class BetafuncEvtVtxGenerator : public BaseEvtVtxGenerator 
{
public:
  BetafuncEvtVtxGenerator(const edm::ParameterSet & p);
  virtual ~BetafuncEvtVtxGenerator();

  /// return a new event vertex
  //virtual CLHEP::Hep3Vector * newVertex();
  virtual HepMC::FourVector* newVertex() ;

  virtual TMatrixD* GetInvLorentzBoost();

    
  /// set resolution in Z in cm
  void sigmaZ(double s=1.0);

  /// set mean in X in cm
  void X0(double m=0) { fX0=m; }
  /// set mean in Y in cm
  void Y0(double m=0) { fY0=m; }
  /// set mean in Z in cm
  void Z0(double m=0) { fZ0=m; }

  /// set half crossing angle
  void Phi(double m=0) { phi_=m; }
  /// angle between crossing plane and horizontal plane
  void Alpha(double m=0) { alpha_=m; }

  /// set beta_star
  void betastar(double m=0) { fbetastar=m; }
  /// emittance (no the normalized)
  void emittance(double m=0) { femittance=m; }

  /// beta function
  double BetaFunction(double z, double z0);
    
private:
  /** Copy constructor */
  BetafuncEvtVtxGenerator(const BetafuncEvtVtxGenerator &p);
  /** Copy assignment operator */
  BetafuncEvtVtxGenerator&  operator = (const BetafuncEvtVtxGenerator & rhs );
  
private:

  double alpha_, phi_;
  //TMatrixD boost_;
  
  double fX0, fY0, fZ0;
  double fSigmaZ;
  //double fdxdz, fdydz;
  double fbetastar, femittance;
  double falpha;
  double fTimeOffset;
    
  CLHEP::RandGaussQ*  fRandom ;
  
};

#endif
