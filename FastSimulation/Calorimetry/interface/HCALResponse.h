#ifndef HCALResponse_h
#define HCALResponse_h

/** \file FastSimulation/Calorimetry/interface/HCALResponse.h. 
 *
 *  Helper class to contain the HCAL response to hadrons and e/gamma
 * 
 */

// CMSSW Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include <map>

#define maxHDet 7
#define maxHDeta 50
#define maxMUet 4
#define maxMUeta 6   
#define maxMUbin 40   

#define maxEMe 5
#define maxEMeta 20

typedef std::pair<double,double> response;
enum part{hcbarrel=0, hcendcap=1, hcforward=2};
enum type{ECAL=0, HCAL=1, VFCAL=2};

class HCALResponse
{
public:
  HCALResponse(const edm::ParameterSet& pset);
  ~HCALResponse(){;} 

  // Get the response in the for of pair 
  // parameters:  energy, eta, e/gamma = 0, hadron = 1, mu = 2
  response responseHCAL(double energy, double eta, int partype);

  // legacy methods using simle furmulae
  double getHCALEnergyResponse   (double e, int hit);
  double getHCALEnergyResolution (double e, int hit);
  double getHFEnergyResolution   (double EGen);
  
private:

  // calculates interpolated-extrapolated reponse (mean and sigma, see below)
  // for hadrons and e/gamma (the latter in HF specifically)
  void interHD(double e, int ie, int ieta);
  void interEM(double e, int ie, int ieta); 
  void interMU(double e, int ie, int ieta); 

  ///Default values for resolution parametrisation:
  ///stochastic, constant and noise.
  //in the barrel and in the endcap
  //in the ECAL, HCAL, VFCAL
  double RespPar[3][2][3];

  ///HCAL response parameters
  double eResponseScale[3];
  double eResponsePlateau[3];
  double eResponseExponent;
  double eResponseCoefficient;
  double eResponseCorrection;
  double eBias;

  // just eta step of the tabulated data
  double etaStep, muStep;
  // mean and sigma
  double mean, sigma; 

  // Tabulated energy, et/pt and eta points
  double etGridHD[maxHDet];    
  double eGridEM[maxEMe];
  double eGridMU[maxMUet];
  double etaGridMU[maxMUeta];

  // Tabulated response and mean for hadrons normalized to the energy 
  double meanHD[maxHDet][maxHDeta], sigmaHD[maxHDet][maxHDeta];  

  // muon histos 
  double responseMU[maxMUet][maxMUeta][maxMUbin]; 

  // Tabulated response and mean for e/gamma in HF specifically (normalized) 
  double meanEM[maxEMe][maxEMeta], sigmaEM[maxEMe][maxEMeta];

};
#endif

