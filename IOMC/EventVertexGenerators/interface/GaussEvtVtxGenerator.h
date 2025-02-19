#ifndef IOMC_GaussEvtVtxGenerator_H
#define IOMC_GaussEvtVtxGenerator_H

/**
 * Generate event vertices according to a Gauss distribution. 
 * Attention: All values are assumed to be cm!
 *
 * $Id: GaussEvtVtxGenerator.h,v 1.6 2008/04/04 21:38:24 yumiceva Exp $
 */

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"

namespace CLHEP {
   class RandGaussQ;
}

class GaussEvtVtxGenerator : public BaseEvtVtxGenerator 
{
public:
  GaussEvtVtxGenerator(const edm::ParameterSet & p);
  virtual ~GaussEvtVtxGenerator();

  /// return a new event vertex
  //virtual CLHEP::Hep3Vector* newVertex();
  virtual HepMC::FourVector* newVertex() ;

  virtual TMatrixD* GetInvLorentzBoost() {
	  return 0;
  }

   
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
  
private:
  /** Copy constructor */
  GaussEvtVtxGenerator(const GaussEvtVtxGenerator &p);
  /** Copy assignment operator */
  GaussEvtVtxGenerator&  operator = (const GaussEvtVtxGenerator & rhs );
private:
  double fSigmaX, fSigmaY, fSigmaZ;
  double fMeanX,  fMeanY,  fMeanZ;
  CLHEP::RandGaussQ*  fRandom ;
  double fTimeOffset;
};

#endif
