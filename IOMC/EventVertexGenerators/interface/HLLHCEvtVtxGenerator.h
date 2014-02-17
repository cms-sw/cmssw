#ifndef IOMC_HLLHCEvtVtxGenerator_H
#define IOMC_HLLHCEvtVtxGenerator_H

/**
 * Generate event vertices given beams sizes, crossing angle
 * offset, and crab rotation. 
 * Attention: All values are assumed to be cm for spatial coordinates
 * and ns for time.
 *
 * $Id: HLLHCEvtVtxGenerator.h,v 1.1 2013/02/08 23:04:38 aryd Exp $
 */

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"

namespace CLHEP {
  class RandGaussQ;
}

namespace edm {
  class ConfigurationDescriptions;
}

class HLLHCEvtVtxGenerator : public BaseEvtVtxGenerator 
{
public:

  HLLHCEvtVtxGenerator(const edm::ParameterSet & p);

  virtual ~HLLHCEvtVtxGenerator();

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  /// return a new event vertex
  virtual HepMC::FourVector* newVertex() ;

  virtual TMatrixD* GetInvLorentzBoost() {
	  return 0;
  }

   
private:
  /** Copy constructor */
  HLLHCEvtVtxGenerator(const HLLHCEvtVtxGenerator &p);

  /** Copy assignment operator */
  HLLHCEvtVtxGenerator&  operator = (const HLLHCEvtVtxGenerator & rhs );
private:

  double fMeanX, fMeanY, fMeanZ;

  double fSigmaX, fSigmaY, fSigmaZ;

  double fHalfCrossingAngle,  fCrabAngle;

  double fTimeOffset;

  CLHEP::RandGaussQ*  fRandom ;

};

#endif
