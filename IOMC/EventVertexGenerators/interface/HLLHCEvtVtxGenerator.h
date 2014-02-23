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

#include <string>

namespace CLHEP {
  class RandFlat;
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

  //spatial and time offset for mean collision
  double fMeanX, fMeanY, fMeanZ, fTimeOffset;

  //proton beam energy
  double fEproton;

  //half crossing angle 
  double fTheta;

  //crab rotation in crossing plane
  double fAlphax;

  //crab frequence in crossing plane
  double fOmegax;

  //normalized emmittance in crossing plane
  double fEpsilonx;

  //beta function in crossing plane
  double fBetax;
  
  //crab rotation in parallel plane
  double fAlphay;

  //crab frequence in parallel plane
  double fOmegay;

  //normalized emmittance parallel plane
  double fEpsilony;

  //beta function in parallel plane
  double fBetay;
  
  //longitudinal bunch size
  double fZsize;

  //longitudinal beam profile
  std::string fProfile;
  // fProfile is one of:
  // "Gaussian" and then fZsize is the width of the gaussian
  // "Flat" and then fZsize is the half length of the bunch.
   
  CLHEP::RandFlat*  fRandom ;

  struct lhcbeamparams {

    double betagamma; 
    double theta;
    double alphax;
    double omegax;
    double epsilonx;
    double betax;
    double alphay;
    double omegay;
    double epsilony;
    double betay;
    double zsize;  
    std::string beamprofile;

  };

  double p1(double x, double y, double z, double t, const lhcbeamparams& par);

  double p2(double x, double y, double z, double t, const lhcbeamparams& par);

  double sigma(double z, double epsilon, double beta, double betagamma);

  double rhoz(double z, const lhcbeamparams& par);

};

#endif
