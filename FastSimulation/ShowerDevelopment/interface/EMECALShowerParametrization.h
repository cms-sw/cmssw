#ifndef EMECALShowerParametrization_H
#define EMECALShowerParametrization_H

#include "FastSimulation/CalorimeterProperties/interface/ECALProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer1Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer2Properties.h"

/** 
 * Electromagnetic Shower parametrization utilities according to 
 * G. Grindhammer and S. Peters, hep-ex/0001020, Appendix A
 *
 * \author Patrick Janot
 * \date: 25-Jan-2004
 */ 
#include <vector>
#include <cmath>

class EMECALShowerParametrization
{
 public:

  EMECALShowerParametrization(const ECALProperties* ecal,
			      const HCALProperties* hcal,
			      const PreshowerLayer1Properties* layer1,
			      const PreshowerLayer2Properties* layer2,
			      const std::vector<double>& coreIntervals,
			      const std::vector<double>& tailIntervals,
			      double RCFact=1.,
			      double RTFact=1.) : 
    theECAL(ecal),  
    theHCAL(hcal),
    theLayer1(layer1), 
    theLayer2(layer2),
    theCore(coreIntervals),
    theTail(tailIntervals),
    theRcfactor(RCFact),
    theRtfactor(RTFact){}

  virtual ~EMECALShowerParametrization() { }


  //====== Longitudinal profiles =======

  // -------- Average -------

  inline double meanT(double lny) const { 
    if (theECAL->isHom()) return meanTHom(lny);
    return meanTSam(lny); }

  inline double meanAlpha(double lny) const { 
    if (theECAL->isHom()) return meanAlphaHom(lny);
    return meanAlphaSam(lny); }


  // Average Homogeneous

  inline double meanTHom(double lny) const { 
    return lny-0.858; }
    //return lny-1.23; }

  inline double meanAlphaHom(double lny) const { 
    return 0.21+(0.492+2.38/theECAL->theZeff())*lny; }


  // Average sampling


  inline double meanTSam(double lny) const { 
    return meanTHom(lny) - 0.59/theECAL->theFs() - 0.53*(1.-theECAL->ehat()); }

  inline double meanAlphaSam(double lny) const { 
    return meanAlphaHom(lny) - 0.444/theECAL->theFs(); }







  // ---- Fluctuated longitudinal profiles ----

  inline double meanLnT(double lny) const {
    if (theECAL->isHom()) return meanLnTHom(lny); 
    return meanLnTSam(lny); }

  inline double sigmaLnT(double lny) const {
    if (theECAL->isHom()) return sigmaLnTHom(lny); 
    return sigmaLnTSam(lny); }


  inline double meanLnAlpha(double lny) const {
    if (theECAL->isHom()) return meanLnAlphaHom(lny); 
    return meanLnAlphaSam(lny); }
  

  inline double sigmaLnAlpha(double lny) const {
    if (theECAL->isHom()) return sigmaLnAlphaHom(lny); 
    return sigmaLnAlphaSam(lny); }


  inline double correlationAlphaT(double lny) const {
    if (theECAL->isHom()) return correlationAlphaTHom(lny); 
    return correlationAlphaTSam(lny); }




  // Fluctuated longitudinal profiles homogeneous

  inline double meanLnTHom(double lny) const {
    return std::log(lny-0.812); }


  inline double sigmaLnTHom(double lny) const {
    return 1./(-1.4+1.26*lny); }

  inline double meanLnAlphaHom(double lny) const {
    return std::log(0.81+(0.458+2.26/theECAL->theZeff())*lny); }

  inline double sigmaLnAlphaHom(double lny) const {
    return 1./(-0.58+0.86*lny); }

  inline double correlationAlphaTHom(double lny) const {
    return 0.705-0.023*lny; }

  // Fluctuated longitudinal profiles sampling


  inline double meanLnTSam(double lny) const {
    return log( std::exp(meanLnTHom(lny)) - 0.55/theECAL->theFs() - 0.69*(1-theECAL->ehat()) ); }

  inline double sigmaLnTSam(double lny) const {
    return 1./(-2.5 + 1.25*lny); }

  inline double meanLnAlphaSam(double lny) const {
    return log( std::exp(meanLnAlphaHom(lny)) - 0.476/theECAL->theFs() ); }

  inline double sigmaLnAlphaSam(double lny) const {
    return 1./(-0.82+0.79*lny); }

  inline double correlationAlphaTSam(double lny) const {
    return 0.784-0.023*lny; }






  //====== Radial profiles =======


  // ---- Radial Profiles ----

  inline double rC(double tau, double E) const {
    if (theECAL->isHom()) return rCHom(tau, E); 
    return rCSam(tau, E); }
  
  inline double rT(double tau, double E) const {
    if (theECAL->isHom()) return rTHom(tau, E); 
    return rTSam(tau, E); }

  inline double p(double tau, double E) const {
    if (theECAL->isHom()) return pHom(tau, E); 
    return pSam(tau, E); }



  // Radial Profiles

  inline double rCHom(double tau, double E) const {
    return theRcfactor*(z1(E) + z2()*tau);
  }  

  inline double rTHom(double tau,double E) const {
    return theRtfactor*k1() * ( std::exp(k3()*(tau-k2()))+
				std::exp(k4(E)*(tau-k2())) );
  }

  inline double pHom(double tau, double E) const {
    double arg = (p2()-tau)/p3(E);
    return p1()* std::exp(arg-std::exp(arg));
  }

  
  // Radial Profiles Sampling

  inline double rCSam(double tau, double E) const {
    return rCHom(tau, E) - 0.0203*(1-theECAL->ehat()) + 0.0397/theECAL->theFs()*std::exp(-1.*tau);
  }

  inline double rTSam(double tau,double E) const {
    return rTHom(tau, E) -0.14*(1-theECAL->ehat()) - 0.495/theECAL->theFs()*std::exp(-1.*tau);
  }

  inline double pSam(double tau, double E) const {
    return pHom(tau, E) + (1-theECAL->ehat())*(0.348-0.642/theECAL->theFs()*std::exp(-1.*std::pow((tau-1),2) ) );
  }






  // ---- Fluctuations of the radial profiles ----

  inline double nSpots(double E) const {
    if (theECAL->isHom()) return nSpotsHom(E); 
    return nSpotsSam(E); }

  inline double meanTSpot(double T) const {
    if (theECAL->isHom()) return meanTSpotHom(T); 
    return meanTSpotSam(T); }

  inline double meanAlphaSpot(double alpha) const {
    if (theECAL->isHom()) return meanAlphaSpotHom(alpha); 
    return meanAlphaSpotSam(alpha); }




  // Fluctuations of the radial profiles
  
  inline double nSpotsHom(double E) const {
    return 93.*std::log(theECAL->theZeff()) * std::pow(E,0.876); }

  inline double meanTSpotHom(double T) const {
    return T*(0.698+0.00212*theECAL->theZeff()); }

  inline double meanAlphaSpotHom(double alpha) const {
    return alpha*(0.639+0.00334*theECAL->theZeff()); }


  // Fluctuations of the radial profiles Sampling

  inline double nSpotsSam(double E) const {
    return 10.3/theECAL->resE()*std::pow(E, 0.959); }

  inline double meanTSpotSam(double T) const {
    return meanTSpotHom(T)*(0.813+0.0019*theECAL->theZeff()); }

  inline double meanAlphaSpotSam(double alpha) const {
    return meanAlphaSpotHom(alpha)*(0.844+0.0026*theECAL->theZeff()); }





  inline const ECALProperties* ecalProperties() const { 
    return theECAL; 
  }

  inline const HCALProperties* hcalProperties() const { 
    return theHCAL; 
  }

  inline const PreshowerLayer1Properties* layer1Properties() const { 
    return theLayer1; 
  }

  inline const PreshowerLayer2Properties* layer2Properties() const { 
    return theLayer2; 
  }

  inline const std::vector<double>& getCoreIntervals() const { return theCore;}

  inline const std::vector<double>& getTailIntervals() const { return theTail;}

 private:
  
  const ECALProperties* theECAL;
  const HCALProperties* theHCAL;
  const PreshowerLayer1Properties* theLayer1;
  const PreshowerLayer2Properties* theLayer2;

  const std::vector<double>& theCore;
  const std::vector<double>& theTail;

  double theRcfactor;
  double theRtfactor;

  double p1() const { return 2.632-0.00094*theECAL->theZeff(); }
  double p2() const { return 0.401+0.00187*theECAL->theZeff(); }
  double p3(double E) const { return 1.313-0.0686*std::log(E); }

  double z1(double E) const { return 0.0251+0.00319*std::log(E); }
  double z2() const { return 0.1162-0.000381*theECAL->theZeff(); }

  double k1() const { return 0.6590-0.00309*theECAL->theZeff(); }
  double k2() const { return 0.6450; }
  double k3() const { return -2.59; }
  double k4(double E) const { return 0.3585+0.0421*std::log(E); }

};

#endif
