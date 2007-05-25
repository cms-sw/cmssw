#ifndef HCALResponse_h
#define HCALResponse_h

/** \file FastSimulation/Calorimetry/interface/HCALResponse.h. 
 *
 *  Helper class to contain the HCAL response to hadrons and e/gamma
 * 
 */

#include <map>

#define maxHDeB   12
#define maxHDeF    7
#define maxEMe     7
#define maxHDetaB 30
#define maxHDetaF 20
#define maxEMeta  20
#define maxMUe     4
#define maxMUeta   6   
#define maxMUbin  40   

typedef std::pair<double,double> response;
enum part{hcbarrel=0, hcendcap=1, hcforward=2};
enum type{ECAL=0, HCAL=1, VFCAL=2};

class RandomEngine;

namespace edm { 
  class ParameterSet;
}

class HCALResponse
{
public:
  HCALResponse(const edm::ParameterSet& pset,
	       const RandomEngine* engine);
  ~HCALResponse(){;} 

  // Get the response in the for of pair 
  // parameters:  energy, eta, e/gamma = 0, hadron = 1, mu = 2
  response responseHCAL(double energy, double eta, int partype);

  // legacy methods using simple furmulae
  double getHCALEnergyResponse   (double e, int hit);
  double getHCALEnergyResolution (double e, int hit);
  double getHFEnergyResolution   (double EGen);
  
private:

  // calculates interpolated-extrapolated reponse (mean and sigma, see below)
  // for hadrons and e/gamma (the latter in HF specifically)
  void interHDB(double e, int ie, int ieta);
  void interHDF(double e, int ie, int ieta);
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
  double eGridHDB[maxHDeB], eGridHDF[maxHDeF];    
  double eGridEM[maxEMe];
  double eGridMU[maxMUe];
  double etaGridMU[maxMUeta];

  // Tabulated response and mean for hadrons normalized to the energy
  double meanHDB[maxHDeB][maxHDetaB], sigmaHDB[maxHDeB][maxHDetaB];  
  double meanHDF[maxHDeF][maxHDetaF], sigmaHDF[maxHDeF][maxHDetaF];  
  // Tabulated response and mean for e/gamma in HF specifically (normalized) 
  double meanEM[maxEMe][maxEMeta], sigmaEM[maxEMe][maxEMeta];

  // muon histos 
  double responseMU[maxMUe][maxMUeta][maxMUbin]; 


  // Famos random engine
  const RandomEngine* random;

};
#endif

